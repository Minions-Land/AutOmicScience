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
      stop(sprintf("参数 %s 缺少值", key))
    }
    value <- args[[idx + 1]]
    res[[substring(key, 3)]] <- value
    idx <- idx + 2
  }
  res
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

ensure_dir <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

load_scdesign3_functions <- function(package_root) {
  if (requireNamespace("scDesign3", quietly = TRUE)) {
    suppressPackageStartupMessages(library(scDesign3))
    message("[INFO] 使用已安装的 scDesign3 包。")
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
    stop("obs.csv 中缺少 cell id 列：", cell_id_col)
  }
  if (!(feature_id_col %in% colnames(var))) {
    stop("var.csv 中缺少 feature id 列：", feature_id_col)
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

`%||%` <- function(a, b) {
  if (is.null(a)) b else a
}

write_metrics <- function(metrics, output_dir) {
  out <- metrics
  jsonlite::write_json(out, file.path(output_dir, "model_metrics.json"), auto_unbox = TRUE, pretty = TRUE)
}

write_count_matrix <- function(mat, path) {
  if (!methods::is(mat, "sparseMatrix")) {
    mat <- Matrix::Matrix(mat, sparse = TRUE)
  }
  Matrix::writeMM(mat, path)
}

write_covariate <- function(df, path) {
  if (is.null(df)) {
    return(invisible(NULL))
  }
  out <- as.data.frame(df)
  out <- cbind(cell_id = rownames(out), out)
  utils::write.csv(out, path, row.names = FALSE)
}

main <- function() {
  args <- parse_args()
  if (is.null(args$config)) {
    stop("必须提供 --config")
  }
  cfg <- jsonlite::fromJSON(args$config, simplifyVector = TRUE)
  package_root <- cfg$package_root
  output_dir <- cfg$output$output_dir
  ensure_dir(output_dir)

  load_scdesign3_functions(package_root)

  sce <- read_input_sce(cfg$prepared_input)

  sim <- cfg$simulation
  sim$celltype <- normalize_null(sim$celltype)
  sim$pseudotime <- normalize_null(sim$pseudotime)
  sim$spatial <- normalize_null(sim$spatial)
  sim$other_covariates <- normalize_null(sim$other_covariates)
  sim$family_use <- sim$family_use %||% "nb"
  sim$assay_use <- sim$assay_use %||% "counts"
  sim$n_cores <- as.integer(sim$n_cores %||% 1)
  sim$ncell <- sim$ncell %||% ncol(sce)
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

  message("[INFO] construct_data 开始")
  dat <- construct_data(
    sce = sce,
    assay_use = sim$assay_use,
    celltype = sim$celltype,
    pseudotime = sim$pseudotime,
    spatial = sim$spatial,
    other_covariates = sim$other_covariates,
    ncell = sim$ncell,
    corr_by = sim$corr_formula
  )

  message("[INFO] fit_marginal 开始")
  marg <- fit_marginal(
    data = dat,
    mu_formula = sim$mu_formula,
    sigma_formula = sim$sigma_formula,
    family_use = sim$family_use,
    n_cores = sim$n_cores,
    usebam = sim$usebam,
    edf_flexible = sim$edf_flexible
  )

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
      n_cores = sim$n_cores
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
      n_cores = sim$n_cores
    )
  }

  message("[INFO] extract_para 开始")
  para <- extract_para(
    sce = sce,
    assay_use = sim$assay_use,
    marginal_list = marg,
    n_cores = sim$n_cores,
    family_use = sim$family_use,
    new_covariate = dat$newCovariate,
    data = dat$dat
  )

  message("[INFO] simu_new 开始")
  if (sim$n_rep == 1) {
    new_count <- simu_new(
      sce = sce,
      assay_use = sim$assay_use,
      mean_mat = para$mean_mat,
      sigma_mat = para$sigma_mat,
      zero_mat = para$zero_mat,
      quantile_mat = if (isTRUE(sim$empirical_quantile)) cop$quantile_mat else NULL,
      copula_list = if (isTRUE(sim$empirical_quantile)) NULL else cop$copula_list,
      n_cores = sim$n_cores,
      fastmvn = sim$fastmvn,
      family_use = sim$family_use,
      nonnegative = sim$nonnegative,
      nonzerovar = sim$nonzerovar,
      input_data = dat$dat,
      new_covariate = dat$newCovariate,
      important_feature = cop$important_feature,
      filtered_gene = dat$filtered_gene
    )
    write_count_matrix(new_count, file.path(output_dir, "sim_counts.mtx"))
    write_covariate(dat$newCovariate, file.path(output_dir, "sim_obs.csv"))
  } else {
    for (rep_idx in seq_len(sim$n_rep)) {
      current_count <- simu_new(
        sce = sce,
        assay_use = sim$assay_use,
        mean_mat = para$mean_mat,
        sigma_mat = para$sigma_mat,
        zero_mat = para$zero_mat,
        quantile_mat = if (isTRUE(sim$empirical_quantile)) cop$quantile_mat else NULL,
        copula_list = if (isTRUE(sim$empirical_quantile)) NULL else cop$copula_list,
        n_cores = sim$n_cores,
        fastmvn = sim$fastmvn,
        family_use = sim$family_use,
        nonnegative = sim$nonnegative,
        nonzerovar = sim$nonzerovar,
        input_data = dat$dat,
        new_covariate = dat$newCovariate,
        important_feature = cop$important_feature,
        filtered_gene = dat$filtered_gene
      )
      write_count_matrix(current_count, file.path(output_dir, sprintf("sim_counts_rep%d.mtx", rep_idx)))
      write_covariate(dat$newCovariate, file.path(output_dir, sprintf("sim_obs_rep%d.csv", rep_idx)))
    }
  }

  if (nrow(SummarizedExperiment::rowData(sce)) > 0) {
    row_meta <- as.data.frame(SummarizedExperiment::rowData(sce))
    row_meta <- cbind(feature_id = rownames(row_meta), row_meta)
    utils::write.csv(row_meta, file.path(output_dir, "sim_var.csv"), row.names = FALSE)
  } else {
    row_meta <- data.frame(feature_id = rownames(sce))
    utils::write.csv(row_meta, file.path(output_dir, "sim_var.csv"), row.names = FALSE)
  }

  metrics <- list(
    model_aic = as.list(cop$model_aic),
    model_bic = as.list(cop$model_bic),
    n_gene = nrow(sce),
    n_cell_input = ncol(sce),
    n_cell_output = if (is.null(dat$newCovariate)) ncol(sce) else nrow(dat$newCovariate),
    filtered_gene = dat$filtered_gene %||% list()
  )
  write_metrics(metrics, output_dir)

  message("[INFO] 输出完成：", output_dir)
}

main()
