#!/usr/bin/env Rscript

options(repos = c(CRAN = "https://cloud.r-project.org"))

cran_packages <- c(
  "BiocManager",
  "remotes",
  "jsonlite",
  "yaml",
  "optparse",
  "Matrix",
  "mgcv",
  "mvtnorm",
  "gamlss",
  "data.table",
  "dplyr",
  "irlba",
  "mclust",
  "pbmcapply"
)

bioc_packages <- c(
  "Biobase",
  "BiocParallel",
  "SingleCellExperiment",
  "SummarizedExperiment",
  "zellkonverter"
)

install_missing_cran <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    install.packages(missing, dependencies = TRUE)
  }
}

install_missing_bioc <- function(packages) {
  install_missing_cran("BiocManager")
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    BiocManager::install(missing, ask = FALSE, update = FALSE)
  }
}

install_missing_cran(cran_packages)
install_missing_bioc(bioc_packages)

if (!requireNamespace("scDesign3", quietly = TRUE)) {
  remotes::install_github("SONGDONGYUAN1994/scDesign3", dependencies = TRUE, upgrade = "never")
}

packages <- c("scDesign3", "SingleCellExperiment", "Matrix", "yaml", "optparse", "zellkonverter")
for (pkg in packages) {
  available <- requireNamespace(pkg, quietly = TRUE)
  version <- if (available) as.character(utils::packageVersion(pkg)) else "missing"
  cat(pkg, version, "\n")
}
