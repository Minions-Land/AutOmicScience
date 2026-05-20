#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
package_root <- if (length(args) >= 1) normalizePath(args[[1]], mustWork = FALSE) else normalizePath("..", mustWork = FALSE)

cran_repos <- getOption("repos")
cran_repos["CRAN"] <- "https://cloud.r-project.org"
options(repos = cran_repos)

install_if_missing <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    install.packages(missing)
  }
}

install_if_missing(c("BiocManager", "remotes"))

bioc_pkgs <- c(
  "SingleCellExperiment",
  "SummarizedExperiment",
  "BiocParallel"
)

missing_bioc <- bioc_pkgs[!vapply(bioc_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_bioc) > 0) {
  BiocManager::install(missing_bioc, ask = FALSE, update = FALSE)
}

if (!dir.exists(package_root)) {
  stop("找不到 package_root: ", package_root)
}

remotes::install_local(
  package_root,
  dependencies = TRUE,
  upgrade = "never",
  force = TRUE
)

message("R 依赖与本地 scDesign3 安装完成。")
