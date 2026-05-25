#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
  library(Matrix)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
})

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  res <- list()
  if (length(args) == 0) {
    stop("请使用 --config <path> 指定配置文件。")
  }
  idx <- 1
  while (idx <= length(args)) {
    key <- args[[idx]]
    if (!startsWith(key, "--")) {
      stop(sprintf("无法解析参数: %s", key))
    }
    if (idx == length(args)) {
      stop(sprintf("参数 %s 缺少取值", key))
    }
    res[[substring(key, 3)]] <- args[[idx + 1]]
    idx <- idx + 2
  }
  res
}

`%||%` <- function(a, b) {
  if (is.null(a)) b else a
}

normalize_null <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  if (length(x) == 1 && is.character(x) && (x %in% c("", "null", "NULL", "None"))) {
    return(NULL)
  }
  x
}

as_character_vec <- function(x) {
  x <- normalize_null(x)
  if (is.null(x)) {
    return(NULL)
  }
  as.character(unlist(x, use.names = FALSE))
}

ensure_dir <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

strip_backticks <- function(x) {
  gsub("`", "", x, fixed = TRUE)
}

quote_name <- function(x) {
  if (grepl("^[A-Za-z.][A-Za-z0-9._]*$", x)) {
    return(x)
  }
  paste0("`", x, "`")
}

load_scdesign3_functions <- function(package_root) {
  local_files <- file.path(package_root, "R", c("fit_copula.R", "extract_para.R", "simu_new.R"))
  if (requireNamespace("scDesign3", quietly = TRUE)) {
    suppressPackageStartupMessages(library(scDesign3))
    message("[INFO] 使用已安装的 scDesign3 包。")
    for (f in local_files[file.exists(local_files)]) {
      source(f, chdir = TRUE)
    }
    message("[INFO] 已加载本地 scDesign3 兼容补丁：", paste(basename(local_files[file.exists(local_files)]), collapse = ", "))
    return(invisible(TRUE))
  }

  message("[INFO] 未检测到已安装 scDesign3，改为 source 本地仓库 R/ 目录。")
  r_dir <- file.path(package_root, "R")
  files <- c(
    "gamlss_fix.R",
    "sparse_cov.R",
    "construct_data.R",
    "fit_marginal.R",
    "fit_copula.R",
    "extract_para.R",
    "simu_new.R",
    "scdesign3.R"
  )
  file_paths <- file.path(r_dir, files)
  missing <- file_paths[!file.exists(file_paths)]
  if (length(missing) > 0) {
    stop("以下源码文件不存在：", paste(missing, collapse = ", "))
  }
  for (f in file_paths) {
    source(f, chdir = TRUE)
  }
  invisible(TRUE)
}

read_standard_bundle_to_sce <- function(bundle_cfg) {
  counts_path <- bundle_cfg$counts_path
  obs_path <- bundle_cfg$obs_path
  var_path <- bundle_cfg$var_path
  orientation <- bundle_cfg$count_orientation %||% "cell_by_gene"
  cell_id_col <- bundle_cfg$cell_id_col %||% "cell_id"
  feature_id_col <- bundle_cfg$feature_id_col %||% "feature_id"

  counts <- Matrix::readMM(counts_path)
  if (orientation == "cell_by_gene") {
    counts <- Matrix::t(counts)
  } else if (orientation != "gene_by_cell") {
    stop("count_orientation 只支持 cell_by_gene 或 gene_by_cell。")
  }

  obs <- data.table::fread(obs_path, data.table = FALSE)
  var <- data.table::fread(var_path, data.table = FALSE)
  if (!(cell_id_col %in% colnames(obs))) {
    stop("obs.csv 缺少 cell id 列：", cell_id_col)
  }
  if (!(feature_id_col %in% colnames(var))) {
    stop("var.csv 缺少 feature id 列：", feature_id_col)
  }

  rownames(obs) <- make.unique(as.character(obs[[cell_id_col]]))
  rownames(var) <- make.unique(as.character(var[[feature_id_col]]))
  obs[[cell_id_col]] <- NULL
  var[[feature_id_col]] <- NULL

  if (ncol(counts) != nrow(obs)) {
    stop(sprintf("counts 细胞数与 obs 行数不一致：%d vs %d", ncol(counts), nrow(obs)))
  }
  if (nrow(counts) != nrow(var)) {
    stop(sprintf("counts 基因数与 var 行数不一致：%d vs %d", nrow(counts), nrow(var)))
  }

  colnames(counts) <- rownames(obs)
  rownames(counts) <- rownames(var)

  SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = counts),
    colData = obs,
    rowData = var
  )
}

read_input_sce <- function(prepared_input) {
  input_type <- prepared_input$type
  if (input_type == "standard_bundle") {
    return(read_standard_bundle_to_sce(prepared_input$bundle))
  }
  if (input_type == "sce_rds") {
    sce <- readRDS(prepared_input$sce_rds_path)
    if (!methods::is(sce, "SingleCellExperiment")) {
      stop("sce_rds_path 读出的对象不是 SingleCellExperiment。")
    }
    return(sce)
  }
  stop("不支持的 prepared_input$type: ", input_type)
}

write_metrics <- function(metrics, output_dir) {
  jsonlite::write_json(metrics, file.path(output_dir, "model_metrics.json"), auto_unbox = TRUE, pretty = TRUE)
}

write_count_matrix <- function(mat, path) {
  if (!methods::is(mat, "sparseMatrix")) {
    mat <- Matrix::Matrix(mat, sparse = TRUE)
  }
  Matrix::writeMM(mat, path)
}

write_covariate <- function(df, path) {
  out <- as.data.frame(df)
  out <- cbind(cell_id = rownames(out), out)
  utils::write.csv(out, path, row.names = FALSE)
}

write_var_metadata <- function(sce, output_dir) {
  if (nrow(SummarizedExperiment::rowData(sce)) > 0) {
    row_meta <- as.data.frame(SummarizedExperiment::rowData(sce))
    row_meta <- cbind(feature_id = rownames(row_meta), row_meta)
    utils::write.csv(row_meta, file.path(output_dir, "sim_var.csv"), row.names = FALSE)
  } else {
    row_meta <- data.frame(feature_id = rownames(sce))
    utils::write.csv(row_meta, file.path(output_dir, "sim_var.csv"), row.names = FALSE)
  }
}

sanitize_sce_colnames <- function(sce) {
  old_names <- colnames(SummarizedExperiment::colData(sce))
  new_names <- make.names(old_names, unique = TRUE)
  colnames(SummarizedExperiment::colData(sce)) <- new_names
  list(sce = sce, name_map = stats::setNames(new_names, old_names))
}

sanitize_name_value <- function(x, name_map) {
  x <- normalize_null(x)
  if (is.null(x)) {
    return(NULL)
  }
  lookup_name <- function(key) {
    key <- as.character(key)
    if (key %in% names(name_map)) {
      return(unname(name_map[[key]]))
    }
    key
  }
  if (length(x) == 1) {
    return(lookup_name(x))
  }
  vapply(as.character(x), lookup_name, character(1))
}

sanitize_formula_value <- function(formula_str, name_map) {
  formula_str <- normalize_null(formula_str)
  if (is.null(formula_str)) {
    return(NULL)
  }
  out <- as.character(formula_str)
  ordered_keys <- names(sort(nchar(names(name_map)), decreasing = TRUE))
  for (old_name in ordered_keys) {
    new_name <- unname(name_map[[old_name]])
    out <- gsub(paste0("`", old_name, "`"), new_name, out, fixed = TRUE)
    out <- gsub(old_name, new_name, out, fixed = TRUE)
  }
  out
}

normalize_simulation_cfg <- function(sim) {
  sim$celltype <- normalize_null(sim$celltype)
  sim$pseudotime <- normalize_null(sim$pseudotime)
  sim$spatial <- normalize_null(sim$spatial)
  sim$other_covariates <- normalize_null(sim$other_covariates)
  sim$family_use <- sim$family_use %||% "nb"
  sim$assay_use <- sim$assay_use %||% "counts"
  sim$n_cores <- as.integer(sim$n_cores %||% 1)
  sim$ncell <- as.integer(sim$ncell)
  sim$corr_formula <- sim$corr_formula %||% "1"
  sim$mu_formula <- sim$mu_formula %||% "1"
  sim$sigma_formula <- sim$sigma_formula %||% "1"
  sim$copula <- sim$copula %||% "gaussian"
  sim$empirical_quantile <- ifelse(is.null(sim$empirical_quantile), FALSE, sim$empirical_quantile)
  sim$DT <- ifelse(is.null(sim$DT), TRUE, sim$DT)
  sim$pseudo_obs <- ifelse(is.null(sim$pseudo_obs), FALSE, sim$pseudo_obs)
  sim$important_feature <- sim$important_feature %||% "all"
  sim$if_sparse <- ifelse(is.null(sim$if_sparse), FALSE, sim$if_sparse)
  sim$fastmvn <- ifelse(is.null(sim$fastmvn), FALSE, sim$fastmvn)
  sim$usebam <- ifelse(is.null(sim$usebam), FALSE, sim$usebam)
  sim$edf_flexible <- ifelse(is.null(sim$edf_flexible), FALSE, sim$edf_flexible)
  sim$nonnegative <- ifelse(is.null(sim$nonnegative), TRUE, sim$nonnegative)
  sim$nonzerovar <- ifelse(is.null(sim$nonzerovar), FALSE, sim$nonzerovar)
  sim$n_rep <- as.integer(sim$n_rep %||% 1)
  sim$parallelization <- sim$parallelization %||% ifelse(.Platform$OS.type == "windows", "bpmapply", "pbmcmapply")
  sim
}

apply_name_map_to_simulation <- function(sim, name_map) {
  sim$celltype <- sanitize_name_value(sim$celltype, name_map)
  sim$pseudotime <- sanitize_name_value(sim$pseudotime, name_map)
  sim$spatial <- sanitize_name_value(sim$spatial, name_map)
  sim$other_covariates <- sanitize_name_value(sim$other_covariates, name_map)
  sim$corr_formula <- sanitize_name_value(sim$corr_formula, name_map)
  sim$mu_formula <- sanitize_formula_value(sim$mu_formula, name_map)
  sim$sigma_formula <- sanitize_formula_value(sim$sigma_formula, name_map)
  sim
}

recompute_corr_group <- function(df, corr_formula) {
  group <- normalize_null(corr_formula)
  if (is.null(group)) {
    stop("corr_formula 不能为空。")
  }
  group <- unlist(group)
  if (length(group) > 1) {
    stop("当前脚本仅支持单列 corr_formula。")
  }
  if (group[1] == "1") {
    df$corr_group <- rep(1, nrow(df))
  } else if (group[1] == "ind") {
    df$corr_group <- rep("ind", nrow(df))
  } else {
    if (!(group[1] %in% colnames(df))) {
      stop("corr_formula 指定的列不存在：", group[1])
    }
    df$corr_group <- df[[group[1]]]
  }
  df
}

restore_factor_levels <- function(df, ref_df, exclude_cols = NULL) {
  common_cols <- setdiff(intersect(colnames(df), colnames(ref_df)), exclude_cols %||% character(0))
  for (col in common_cols) {
    if (is.factor(ref_df[[col]])) {
      df[[col]] <- factor(as.character(df[[col]]), levels = levels(ref_df[[col]]))
    }
  }
  df
}

validate_marginal_list <- function(marginal_list) {
  failed <- vapply(
    marginal_list,
    function(x) is.null(x$fit) || length(x$fit) == 0 || (length(x$fit) == 1 && is.atomic(x$fit) && is.na(x$fit)),
    logical(1)
  )
  failed_n <- sum(failed)
  if (failed_n > 0) {
    message("[WARN] fit_marginal 中有 ", failed_n, " 个基因拟合失败。")
  }
  if (failed_n == length(marginal_list)) {
    stop("fit_marginal 所有基因均失败，无法继续。请减小拟合复杂度或检查输入数据。")
  }
  marginal_list
}

align_covariate_to_input_data <- function(new_covariate, input_data) {
  required_cols <- colnames(input_data)
  missing_cols <- setdiff(required_cols, colnames(new_covariate))
  if (length(missing_cols) > 0) {
    stop("new_covariate 缺少 input_data 所需列：", paste(missing_cols, collapse = ", "))
  }
  aligned <- new_covariate[, required_cols, drop = FALSE]
  rownames(aligned) <- rownames(new_covariate)
  aligned
}

fit_master_model <- function(sce, sim) {
  message("[INFO] construct_data 开始")
  dat <- construct_data(
    sce = sce,
    assay_use = sim$assay_use,
    celltype = sim$celltype,
    pseudotime = sim$pseudotime,
    spatial = sim$spatial,
    other_covariates = sim$other_covariates,
    ncell = sim$ncell,
    corr_by = sim$corr_formula,
    parallelization = sim$parallelization
  )

  message("[INFO] fit_marginal 开始")
  marg <- fit_marginal(
    data = dat,
    mu_formula = sim$mu_formula,
    sigma_formula = sim$sigma_formula,
    family_use = sim$family_use,
    n_cores = sim$n_cores,
    usebam = sim$usebam,
    edf_flexible = sim$edf_flexible,
    parallelization = sim$parallelization,
    simplify = FALSE
  )
  marg <- validate_marginal_list(marg)

  message("[INFO] fit_copula 开始")
  if (isTRUE(sim$empirical_quantile)) {
    cop <- fit_copula(
      sce = sce,
      assay_use = sim$assay_use,
      input_data = dat$dat,
      empirical_quantile = TRUE,
      marginal_list = marg,
      family_use = sim$family_use,
      copula = sim$copula,
      important_feature = sim$important_feature,
      if_sparse = sim$if_sparse,
      n_cores = sim$n_cores,
      parallelization = sim$parallelization
    )
  } else {
    cop <- fit_copula(
      sce = sce,
      assay_use = sim$assay_use,
      input_data = dat$dat,
      marginal_list = marg,
      family_use = sim$family_use,
      copula = sim$copula,
      DT = sim$DT,
      pseudo_obs = sim$pseudo_obs,
      important_feature = sim$important_feature,
      if_sparse = sim$if_sparse,
      n_cores = sim$n_cores,
      parallelization = sim$parallelization
    )
  }

  list(dat = dat, marginal = marg, copula = cop)
}

extract_para_with_covariate <- function(sce, sim, dat, marginal_list, new_covariate) {
  message("[INFO] extract_para 开始")
  model_covariate <- align_covariate_to_input_data(new_covariate, dat$dat)
  extract_para(
    sce = sce,
    assay_use = sim$assay_use,
    marginal_list = marginal_list,
    n_cores = sim$n_cores,
    family_use = sim$family_use,
    new_covariate = model_covariate,
    parallelization = sim$parallelization,
    data = dat$dat
  )
}

build_group_design <- function(covariate_df, modeled_cols) {
  use_df <- as.data.frame(covariate_df[, modeled_cols, drop = FALSE])
  for (col in modeled_cols) {
    if (is.character(use_df[[col]])) {
      use_df[[col]] <- factor(use_df[[col]])
    }
    if (!(is.factor(use_df[[col]]) || is.logical(use_df[[col]]))) {
      stop("当前 effect scaling 仅支持分类协变量，列不满足要求：", col)
    }
  }

  group_key <- interaction(use_df, drop = TRUE, lex.order = TRUE, sep = "___scd3___")
  group_levels <- levels(group_key)
  rep_idx <- match(group_levels, group_key)
  group_df <- use_df[rep_idx, , drop = FALSE]
  weights <- as.numeric(table(group_key))

  formula_str <- paste("~", paste(vapply(modeled_cols, quote_name, character(1)), collapse = " + "))
  design_formula <- stats::as.formula(formula_str)
  design <- stats::model.matrix(design_formula, data = group_df)
  term_labels <- strip_backticks(attr(stats::terms(design_formula), "term.labels"))
  assign <- attr(design, "assign")

  list(
    group_key = group_key,
    group_levels = group_levels,
    rep_idx = rep_idx,
    design = design,
    weights = weights,
    term_labels = term_labels,
    assign = assign
  )
}

scale_target_effect_in_mean_mat <- function(mean_mat, covariate_df, modeled_cols, target_col, scale_factor) {
  if (!(target_col %in% modeled_cols)) {
    stop("target_col 不在 modeled_cols 中：", target_col)
  }
  design_info <- build_group_design(covariate_df, modeled_cols)
  target_term_idx <- which(design_info$term_labels == target_col)
  if (length(target_term_idx) != 1) {
    stop("无法在设计矩阵中唯一定位目标项：", target_col)
  }
  target_coef_cols <- which(design_info$assign == target_term_idx)
  if (length(target_coef_cols) == 0) {
    stop("目标项没有可缩放的系数列：", target_col)
  }

  old_eta_group <- log(pmax(mean_mat[design_info$rep_idx, , drop = FALSE], 1e-8))
  new_eta_group <- matrix(NA_real_, nrow = nrow(old_eta_group), ncol = ncol(old_eta_group))

  for (gene_idx in seq_len(ncol(old_eta_group))) {
    fit <- stats::lm.wfit(
      x = design_info$design,
      y = old_eta_group[, gene_idx],
      w = design_info$weights
    )
    coef <- fit$coefficients
    coef[is.na(coef)] <- 0
    coef[target_coef_cols] <- coef[target_coef_cols] * scale_factor
    new_eta_group[, gene_idx] <- as.vector(design_info$design %*% coef)
  }

  ratio_group <- exp(new_eta_group - old_eta_group)
  row_groups <- split(seq_len(nrow(covariate_df)), design_info$group_key)
  for (group_idx in seq_along(design_info$group_levels)) {
    rows <- row_groups[[design_info$group_levels[[group_idx]]]]
    mean_mat[rows, ] <- sweep(mean_mat[rows, , drop = FALSE], 2, ratio_group[group_idx, ], "*")
  }
  pmax(mean_mat, 1e-8)
}

build_virtual_donor_template <- function(
  df,
  donor_col,
  donor_ids,
  target_total,
  seed = 1L,
  anchor_backup_col = NULL,
  anchor_value = NULL
) {
  donor_ids <- as.character(donor_ids)
  if (length(donor_ids) == 0) {
    stop("generation_template.virtual_donor_ids 不能为空")
  }
  target_total <- as.integer(target_total)
  if (target_total <= 0) {
    stop("generation_template.target_total 必须 > 0")
  }

  base_n <- target_total %/% length(donor_ids)
  remainder <- target_total %% length(donor_ids)
  donor_counts <- rep(base_n, length(donor_ids))
  if (remainder > 0) {
    donor_counts[seq_len(remainder)] <- donor_counts[seq_len(remainder)] + 1L
  }

  set.seed(as.integer(seed))
  out_list <- vector("list", length(donor_ids))
  for (idx in seq_along(donor_ids)) {
    sampled_idx <- sample(seq_len(nrow(df)), size = donor_counts[[idx]], replace = TRUE)
    donor_df <- df[sampled_idx, , drop = FALSE]
    if (!is.null(anchor_backup_col)) {
      if (donor_col %in% colnames(donor_df)) {
        donor_df[[anchor_backup_col]] <- as.character(donor_df[[donor_col]])
      } else {
        donor_df[[anchor_backup_col]] <- rep(anchor_value %||% NA_character_, nrow(donor_df))
      }
    }
    donor_df[[donor_col]] <- donor_ids[[idx]]
    out_list[[idx]] <- donor_df
  }
  out <- do.call(rbind, out_list)
  if (donor_col %in% colnames(df) && is.factor(df[[donor_col]])) {
    out[[donor_col]] <- factor(as.character(out[[donor_col]]), levels = donor_ids)
  }
  rownames(out) <- paste0("Cell", seq_len(nrow(out)))
  out
}

resample_covariate_for_target_prop_by_donor <- function(
  df,
  donor_col,
  celltype_col,
  target_celltype,
  target_prop,
  seed = 1L
) {
  donor_values <- as.character(df[[donor_col]])
  donors <- unique(donor_values)
  group_sizes <- table(donor_values)
  total_n <- nrow(df)
  target_total <- as.integer(round(total_n * target_prop))
  if (target_total <= 0) {
    stop("目标稀有细胞数为 0，请提高 target_prop 或总细胞数")
  }

  base_counts <- floor(as.numeric(group_sizes) / total_n * target_total)
  remainder <- target_total - sum(base_counts)
  if (remainder > 0) {
    frac <- as.numeric(group_sizes) / total_n * target_total - base_counts
    add_idx <- order(frac, decreasing = TRUE)[seq_len(remainder)]
    base_counts[add_idx] <- base_counts[add_idx] + 1L
  }
  names(base_counts) <- names(group_sizes)

  set.seed(as.integer(seed))
  out_list <- vector("list", length(donors))
  for (idx in seq_along(donors)) {
    donor <- donors[[idx]]
    donor_idx <- which(donor_values == donor)
    donor_df <- df[donor_idx, , drop = FALSE]
    group_n <- nrow(donor_df)
    target_n <- as.integer(base_counts[[donor]])
    target_pool <- which(as.character(donor_df[[celltype_col]]) == target_celltype)
    other_pool <- which(as.character(donor_df[[celltype_col]]) != target_celltype)
    if (length(target_pool) == 0) {
      stop("donor 中不存在目标细胞类型，无法构造稀有细胞：", donor, " / ", target_celltype)
    }
    if (length(other_pool) == 0) {
      stop("donor 中只有目标细胞类型，无法构造稀有细胞：", donor, " / ", target_celltype)
    }
    sampled_target <- donor_df[sample(target_pool, size = target_n, replace = TRUE), , drop = FALSE]
    sampled_other <- donor_df[sample(other_pool, size = group_n - target_n, replace = TRUE), , drop = FALSE]
    donor_out <- rbind(sampled_target, sampled_other)
    donor_out <- donor_out[sample(seq_len(nrow(donor_out))), , drop = FALSE]
    out_list[[idx]] <- donor_out
  }
  out <- do.call(rbind, out_list)
  rownames(out) <- rownames(df)
  out
}

remove_celltypes_from_donors <- function(
  df,
  donor_col,
  celltype_col,
  donors,
  missing_celltypes,
  protected_cols = NULL,
  seed = 1L
) {
  out <- df
  missing_celltypes <- as.character(missing_celltypes)
  protected_cols <- unique(c(protected_cols %||% character(0), "corr_group"))
  replace_cols <- setdiff(colnames(out), protected_cols)
  set.seed(as.integer(seed))

  for (donor in donors) {
    donor_idx <- which(as.character(out[[donor_col]]) == donor)
    if (length(donor_idx) == 0) {
      warning("donor 不存在，跳过：", donor)
      next
    }
    donor_types <- as.character(out[[celltype_col]][donor_idx])
    target_idx <- donor_idx[donor_types %in% missing_celltypes]
    if (length(target_idx) == 0) {
      warning("donor 中本就不存在目标细胞类型，跳过：", donor)
      next
    }
    replacement_pool <- donor_idx[!(donor_types %in% missing_celltypes)]
    if (length(replacement_pool) == 0) {
      stop("donor 中除目标细胞类型外没有其他细胞，无法替换：", donor)
    }
    replacement_idx <- sample(replacement_pool, size = length(target_idx), replace = TRUE)
    out[target_idx, replace_cols] <- out[replacement_idx, replace_cols]
    out[target_idx, donor_col] <- donor
  }
  out
}

apply_virtual_batch_effect <- function(
  mean_mat,
  donor_values,
  gene_names,
  gene_fraction,
  log_shift_sd,
  global_scale_sd,
  seed = 1L
) {
  donor_values <- as.character(donor_values)
  donors <- unique(donor_values)
  n_gene <- ncol(mean_mat)
  affected_n <- max(1L, as.integer(round(n_gene * gene_fraction)))
  set.seed(as.integer(seed))
  donor_summary <- vector("list", length(donors))

  for (idx in seq_along(donors)) {
    donor <- donors[[idx]]
    rows <- which(donor_values == donor)
    global_shift <- stats::rnorm(1, mean = 0, sd = global_scale_sd)
    gene_idx <- sample(seq_len(n_gene), size = affected_n, replace = FALSE)
    gene_shift <- rep(0, n_gene)
    gene_shift[gene_idx] <- stats::rnorm(length(gene_idx), mean = 0, sd = log_shift_sd)
    scale_vec <- exp(global_shift + gene_shift)
    mean_mat[rows, ] <- sweep(mean_mat[rows, , drop = FALSE], 2, scale_vec, "*")
    donor_summary[[idx]] <- list(
      donor = donor,
      global_log_shift = unname(global_shift),
      n_affected_genes = length(gene_idx),
      affected_genes_preview = as.character(gene_names[gene_idx[seq_len(min(10, length(gene_idx)))]]),
      mean_scale_preview = as.numeric(scale_vec[gene_idx[seq_len(min(10, length(gene_idx)))]])
    )
  }

  list(mean_mat = pmax(mean_mat, 1e-8), donor_summary = donor_summary)
}

simulate_and_write_variant <- function(
  variant_dir,
  variant_name,
  sce,
  sim,
  dat,
  cop,
  para,
  new_covariate,
  variant_meta
) {
  ensure_dir(variant_dir)
  message("[INFO] simu_new 开始：", variant_name)
  model_covariate <- align_covariate_to_input_data(new_covariate, dat$dat)
  call_simu_new_once <- function(use_fastmvn) {
    simu_new(
      sce = sce,
      assay_use = sim$assay_use,
      mean_mat = para$mean_mat,
      sigma_mat = para$sigma_mat,
      zero_mat = para$zero_mat,
      quantile_mat = if (isTRUE(sim$empirical_quantile)) cop$quantile_mat else NULL,
      copula_list = if (isTRUE(sim$empirical_quantile)) NULL else cop$copula_list,
      n_cores = sim$n_cores,
      fastmvn = use_fastmvn,
      family_use = sim$family_use,
      nonnegative = sim$nonnegative,
      nonzerovar = sim$nonzerovar,
      input_data = dat$dat,
      new_covariate = model_covariate,
      important_feature = cop$important_feature,
      parallelization = sim$parallelization,
      filtered_gene = dat$filtered_gene
    )
  }
  call_simu_new_safe <- function() {
    tryCatch(
      call_simu_new_once(sim$fastmvn),
      error = function(e) {
        err_msg <- conditionMessage(e)
        if (isTRUE(sim$fastmvn) && grepl("chol\\(", err_msg, fixed = FALSE)) {
          message("[WARN] 检测到 fastmvn 的 chol 分解失败，自动回退到 eigen 采样（fastmvn = FALSE）。")
          return(call_simu_new_once(FALSE))
        }
        stop(e)
      }
    )
  }
  if (sim$n_rep == 1) {
    new_count <- call_simu_new_safe()
    write_count_matrix(new_count, file.path(variant_dir, "sim_counts.mtx"))
    write_covariate(new_covariate, file.path(variant_dir, "sim_obs.csv"))
  } else {
    for (rep_idx in seq_len(sim$n_rep)) {
      current_count <- call_simu_new_safe()
      write_count_matrix(current_count, file.path(variant_dir, sprintf("sim_counts_rep%d.mtx", rep_idx)))
      write_covariate(new_covariate, file.path(variant_dir, sprintf("sim_obs_rep%d.csv", rep_idx)))
    }
  }

  write_var_metadata(sce, variant_dir)
  metrics <- list(
    model_aic = as.list(cop$model_aic),
    model_bic = as.list(cop$model_bic),
    n_gene = nrow(sce),
    n_cell_input = ncol(sce),
    n_cell_output = nrow(new_covariate),
    filtered_gene = dat$filtered_gene %||% list(),
    variant = variant_meta
  )
  write_metrics(metrics, variant_dir)
  jsonlite::write_json(variant_meta, file.path(variant_dir, "variant_summary.json"), auto_unbox = TRUE, pretty = TRUE)
}

resolve_selected_variants <- function(variants_cfg, selected_variants) {
  if (!is.null(selected_variants) && length(selected_variants) > 0) {
    return(as.character(selected_variants))
  }
  out <- character(0)
  if (is.null(variants_cfg$baseline) || isTRUE(variants_cfg$baseline$enabled)) out <- c(out, "baseline")
  if (!is.null(variants_cfg$variant1) && isTRUE(variants_cfg$variant1$enabled)) out <- c(out, "variant1")
  if (!is.null(variants_cfg$variant2) && isTRUE(variants_cfg$variant2$enabled)) out <- c(out, "variant2")
  if (!is.null(variants_cfg$variant3) && isTRUE(variants_cfg$variant3$enabled)) out <- c(out, "variant3")
  if (!is.null(variants_cfg$variant4) && isTRUE(variants_cfg$variant4$enabled)) out <- c(out, "variant4")
  out
}

run_selected_variants <- function(cache, cfg, output_dir) {
  sce <- cache$sce
  sim <- cache$sim
  dat <- cache$fit$dat
  marg <- cache$fit$marginal
  cop <- cache$fit$copula
  name_map <- cache$name_map
  gene_names <- rownames(sce)

  anchor_cfg <- cfg$anchor_donor %||% list()
  variants_cfg <- cfg$variants %||% list()
  template_cfg <- cfg$generation_template %||% list()
  selected_variants <- resolve_selected_variants(variants_cfg, cfg$selected_variants)
  if (length(selected_variants) == 0) {
    stop("没有需要运行的变体。请检查 variants.enabled 或 --variants 参数。")
  }
  message("[INFO] 计划生成的变体：", paste(selected_variants, collapse = ", "))

  donor_col <- sanitize_name_value(template_cfg$virtual_donor_column %||% (anchor_cfg$column %||% "Donor ID"), name_map)
  donor_col <- donor_col %||% "Donor.ID"
  backup_col <- template_cfg$anchor_backup_column %||% "anchor_donor_id"
  donor_ids <- as_character_vec(template_cfg$virtual_donor_ids)
  if (is.null(donor_ids) || length(donor_ids) == 0) {
    donor_ids <- sprintf("VDonor%02d", seq_len(as.integer(template_cfg$n_virtual_donors %||% 10L)))
  }
  target_total <- as.integer(template_cfg$target_total %||% 100000L)
  template_cov <- build_virtual_donor_template(
    df = dat$dat,
    donor_col = donor_col,
    donor_ids = donor_ids,
    target_total = target_total,
    seed = template_cfg$seed %||% 1L,
    anchor_backup_col = backup_col,
    anchor_value = anchor_cfg$value %||% cache$anchor_donor
  )
  template_cov <- restore_factor_levels(template_cov, dat$dat, exclude_cols = c(donor_col, backup_col))
  template_cov <- recompute_corr_group(template_cov, sim$corr_formula)

  manifest <- list()

  if ("baseline" %in% selected_variants) {
    variant_dir_name <- "baseline_anchor_100k"
    para <- extract_para_with_covariate(sce, sim, dat, marg, template_cov)
    variant_meta <- list(
      variant_name = variant_dir_name,
      description = "Baseline：Anchor Donor 原始参数 + 10 个虚拟 donor 模板，不注入 batch shift。",
      task = c("annotation", "integration"),
      anchor_donor = cache$anchor_donor,
      virtual_donor_column = donor_col,
      virtual_donor_ids = donor_ids,
      total_ncell = nrow(template_cov)
    )
    simulate_and_write_variant(file.path(output_dir, variant_dir_name), variant_dir_name, sce, sim, dat, cop, para, template_cov, variant_meta)
    manifest[[length(manifest) + 1]] <- list(name = variant_dir_name, dir = variant_dir_name, n_rep = sim$n_rep)
  }

  if ("variant1" %in% selected_variants) {
    variant_dir_name <- "variant1_signal80_100k"
    para <- extract_para_with_covariate(sce, sim, dat, marg, template_cov)
    para$mean_mat <- scale_target_effect_in_mean_mat(
      mean_mat = para$mean_mat,
      covariate_df = template_cov,
      modeled_cols = sim$celltype,
      target_col = sim$celltype,
      scale_factor = variants_cfg$variant1$scale_factor %||% 0.8
    )
    variant_meta <- list(
      variant_name = variant_dir_name,
      description = "方案一：细胞类型均值效应缩小 20%，测试细粒度注释能力。",
      task = "annotation",
      anchor_donor = cache$anchor_donor,
      celltype_column = sim$celltype,
      scale_factor = variants_cfg$variant1$scale_factor %||% 0.8,
      total_ncell = nrow(template_cov)
    )
    simulate_and_write_variant(file.path(output_dir, variant_dir_name), variant_dir_name, sce, sim, dat, cop, para, template_cov, variant_meta)
    manifest[[length(manifest) + 1]] <- list(name = variant_dir_name, dir = variant_dir_name, n_rep = sim$n_rep)
  }

  if ("variant2" %in% selected_variants) {
    variant_dir_name <- "variant2_rare0p5pct_100k"
    rare_celltype <- variants_cfg$variant2$rare_celltype
    if (is.null(rare_celltype)) {
      stop("variant2 必须提供 rare_celltype")
    }
    new_cov <- resample_covariate_for_target_prop_by_donor(
      df = template_cov,
      donor_col = donor_col,
      celltype_col = sim$celltype,
      target_celltype = rare_celltype,
      target_prop = variants_cfg$variant2$rare_proportion %||% 0.005,
      seed = variants_cfg$variant2$seed %||% 1L
    )
    new_cov <- restore_factor_levels(new_cov, template_cov, exclude_cols = c(donor_col, backup_col))
    new_cov <- recompute_corr_group(new_cov, sim$corr_formula)
    para <- extract_para_with_covariate(sce, sim, dat, marg, new_cov)
    target_count <- sum(as.character(new_cov[[sim$celltype]]) == rare_celltype)
    variant_meta <- list(
      variant_name = variant_dir_name,
      description = "方案二：将目标细胞类型压到 0.5%，测试稀有细胞灵敏度。",
      task = "annotation",
      anchor_donor = cache$anchor_donor,
      celltype_column = sim$celltype,
      rare_celltype = rare_celltype,
      rare_proportion = variants_cfg$variant2$rare_proportion %||% 0.005,
      actual_target_count = target_count,
      total_ncell = nrow(new_cov)
    )
    simulate_and_write_variant(file.path(output_dir, variant_dir_name), variant_dir_name, sce, sim, dat, cop, para, new_cov, variant_meta)
    manifest[[length(manifest) + 1]] <- list(name = variant_dir_name, dir = variant_dir_name, n_rep = sim$n_rep)
  }

  if ("variant3" %in% selected_variants) {
    variant_dir_name <- "variant3_virtual_batch_100k"
    para <- extract_para_with_covariate(sce, sim, dat, marg, template_cov)
    batch_res <- apply_virtual_batch_effect(
      mean_mat = para$mean_mat,
      donor_values = template_cov[[donor_col]],
      gene_names = gene_names,
      gene_fraction = variants_cfg$variant3$gene_fraction %||% 0.2,
      log_shift_sd = variants_cfg$variant3$log_shift_sd %||% 0.15,
      global_scale_sd = variants_cfg$variant3$global_scale_sd %||% 0.08,
      seed = variants_cfg$variant3$seed %||% 1L
    )
    para$mean_mat <- batch_res$mean_mat
    variant_meta <- list(
      variant_name = variant_dir_name,
      description = "方案三：对 10 个虚拟 donor 注入人工 batch shift，测试整合鲁棒性。",
      task = "integration",
      anchor_donor = cache$anchor_donor,
      donor_column = donor_col,
      donor_effect_summary = batch_res$donor_summary,
      gene_fraction = variants_cfg$variant3$gene_fraction %||% 0.2,
      log_shift_sd = variants_cfg$variant3$log_shift_sd %||% 0.15,
      global_scale_sd = variants_cfg$variant3$global_scale_sd %||% 0.08,
      total_ncell = nrow(template_cov)
    )
    simulate_and_write_variant(file.path(output_dir, variant_dir_name), variant_dir_name, sce, sim, dat, cop, para, template_cov, variant_meta)
    manifest[[length(manifest) + 1]] <- list(name = variant_dir_name, dir = variant_dir_name, n_rep = sim$n_rep)
  }

  if ("variant4" %in% selected_variants) {
    variant_dir_name <- "variant4_missing_celltypes_100k"
    target_donors <- as_character_vec(variants_cfg$variant4$donors)
    missing_celltypes <- as_character_vec(variants_cfg$variant4$missing_celltypes)
    if (is.null(target_donors) || length(target_donors) == 0) {
      stop("variant4 必须提供 donors")
    }
    if (is.null(missing_celltypes) || length(missing_celltypes) == 0) {
      if (!is.null(variants_cfg$variant4$missing_celltype)) {
        missing_celltypes <- as.character(variants_cfg$variant4$missing_celltype)
      } else {
        stop("variant4 必须提供 missing_celltypes 或 missing_celltype")
      }
    }
    new_cov <- remove_celltypes_from_donors(
      df = template_cov,
      donor_col = donor_col,
      celltype_col = sim$celltype,
      donors = target_donors,
      missing_celltypes = missing_celltypes,
      protected_cols = c(donor_col, backup_col),
      seed = variants_cfg$variant4$seed %||% 1L
    )
    new_cov <- restore_factor_levels(new_cov, template_cov, exclude_cols = c(donor_col, backup_col))
    new_cov <- recompute_corr_group(new_cov, sim$corr_formula)
    para <- extract_para_with_covariate(sce, sim, dat, marg, new_cov)
    batch_res <- apply_virtual_batch_effect(
      mean_mat = para$mean_mat,
      donor_values = new_cov[[donor_col]],
      gene_names = gene_names,
      gene_fraction = variants_cfg$variant4$gene_fraction %||% (variants_cfg$variant3$gene_fraction %||% 0.2),
      log_shift_sd = variants_cfg$variant4$log_shift_sd %||% (variants_cfg$variant3$log_shift_sd %||% 0.15),
      global_scale_sd = variants_cfg$variant4$global_scale_sd %||% (variants_cfg$variant3$global_scale_sd %||% 0.08),
      seed = variants_cfg$variant4$batch_seed %||% 11L
    )
    para$mean_mat <- batch_res$mean_mat
    donor_summary <- lapply(target_donors, function(donor) {
      donor_idx <- which(as.character(new_cov[[donor_col]]) == donor)
      remaining <- sapply(missing_celltypes, function(celltype) {
        sum(as.character(new_cov[[sim$celltype]][donor_idx]) == celltype)
      })
      list(donor = donor, remaining_missing_targets = as.list(stats::setNames(as.integer(remaining), missing_celltypes)))
    })
    variant_meta <- list(
      variant_name = variant_dir_name,
      description = "方案四：在人工 batch shift 基础上，让部分虚拟 donor 缺失若干细胞类型，测试 over-correction。",
      task = "integration",
      anchor_donor = cache$anchor_donor,
      donor_column = donor_col,
      donors = target_donors,
      missing_celltypes = missing_celltypes,
      donor_missing_summary = donor_summary,
      donor_effect_summary = batch_res$donor_summary,
      total_ncell = nrow(new_cov)
    )
    simulate_and_write_variant(file.path(output_dir, variant_dir_name), variant_dir_name, sce, sim, dat, cop, para, new_cov, variant_meta)
    manifest[[length(manifest) + 1]] <- list(name = variant_dir_name, dir = variant_dir_name, n_rep = sim$n_rep)
  }

  jsonlite::write_json(
    list(created_at = as.character(Sys.time()), variants = manifest),
    file.path(output_dir, "variant_manifest.json"),
    auto_unbox = TRUE,
    pretty = TRUE
  )
}

main <- function() {
  args <- parse_args()
  if (is.null(args$config)) {
    stop("必须提供 --config")
  }

  cfg <- jsonlite::fromJSON(args$config, simplifyVector = TRUE)
  output_dir <- cfg$output$output_dir
  ensure_dir(output_dir)
  cache_path <- cfg$output$master_cache_path %||% file.path(output_dir, "anchor_master_model.rds")

  load_scdesign3_functions(cfg$package_root)

  if (isTRUE(cfg$output$force_refit) && file.exists(cache_path)) {
    unlink(cache_path)
  }

  if (file.exists(cache_path)) {
    message("[INFO] 检测到 master cache，直接读取：", cache_path)
    cache <- readRDS(cache_path)
  } else {
    message("[INFO] 未检测到 master cache，开始 Anchor Donor 一次性拟合。")
    sce <- read_input_sce(cfg$prepared_input)
    sanitized <- sanitize_sce_colnames(sce)
    sce <- sanitized$sce
    name_map <- sanitized$name_map
    sim <- normalize_simulation_cfg(cfg$simulation)
    sim <- apply_name_map_to_simulation(sim, name_map)
    if (is.null(sim$ncell) || length(sim$ncell) == 0 || is.na(sim$ncell)) {
      sim$ncell <- ncol(sce)
    }
    anchor_value <- cfg$anchor_donor$value
    if (is.null(anchor_value)) {
      donor_column <- sanitize_name_value(cfg$anchor_donor$column %||% "Donor ID", name_map)
      if (!is.null(donor_column) && donor_column %in% colnames(SummarizedExperiment::colData(sce))) {
        anchor_value <- unique(as.character(SummarizedExperiment::colData(sce)[[donor_column]]))[[1]]
      } else {
        anchor_value <- "anchor_donor"
      }
    }
    fit <- fit_master_model(sce, sim)
    cache <- list(sce = sce, sim = sim, fit = fit, name_map = name_map, anchor_donor = anchor_value)
    saveRDS(cache, cache_path)
    message("[INFO] master cache 已保存：", cache_path)
  }

  run_selected_variants(cache, cfg, output_dir)
  message("[INFO] Anchor Donor 四变体流程完成：", output_dir)
}

main()
