library(randtoolbox)
library(dplyr)
library(DiceKriging)
library(DiceOptim)
library(rsm)

quiet <- function(x) {
  sink(tempfile())
  on.exit(sink())
  invisible(force(x))
}

starting_sobol_n <- 216

sobol_n <- 3000
sobol_dim <- 54 * 2
bit_min <- 1
bit_max <- 8

temp_sobol <- sobol(n = sobol_n,
                    dim = sobol_dim,
                    scrambling = 3,
                    seed = as.integer((99999 - 10000) * runif(1) + 10000),
                    init = TRUE)

rm(temp_sobol)
quiet(gc())

design <- sobol(n = sobol_n,
                dim = sobol_dim,
                scrambling = 3,
                seed = as.integer((99999 - 10000) * runif(1) + 10000),
                init = FALSE)

df_design <- data.frame(design)

names(df_design) <- c(rbind(paste("W", seq(1:(sobol_dim / 2)), sep = ""), paste("A", seq(1:(sobol_dim / 2)), sep = "")))

formulas <- character(length(names(df_design)))

for(i in 1:length(names(df_design))){
  formulas[i] <- paste(names(df_design)[i],
                       "e ~ round((",
                       bit_max - bit_min,
                       " * ",
                       names(df_design)[i],
                       ") + ",
                       bit_min,
                       ")",
                       sep = "")
}

#coded_design <- coded.data(df_design, formulas = lapply(formulas, formula))
coded_design <- df_design

#coded_df_design <- round(data.frame(coded_design))
coded_df_design <- data.frame(coded_design)

write.csv(coded_df_design, paste("sobol_resnet50_",
                                 "weight_activation_",
                                 sobol_n,
                                 "_samples.csv",
                                 sep = ""),
          row.names = FALSE)
