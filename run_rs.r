library(randtoolbox)
library(dplyr)
library(rsm)

quiet <- function(x) {
  sink(tempfile())
  on.exit(sink())
  invisible(force(x))
}

iterations <- 10

search_space <- NULL
results <- NULL

sobol_dim <- 54 * 2
starting_sobol_n <- 2 * sobol_dim

sobol_n <- starting_sobol_n

bit_min <- 1
bit_max <- 8

for(i in 1:iterations){
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

    names(df_design) <- c(rbind(paste("W",
                                      seq(1:(sobol_dim / 2)),
                                      sep = ""),
                                paste("A",
                                      seq(1:(sobol_dim / 2)),
                                      sep = "")))

    coded_design <- df_design

    coded_df_design <- data.frame(coded_design)

    write.csv(coded_df_design, "current_design.csv", row.names = FALSE)

    start_time <- Sys.time()

    cmd <- paste("python3 -W ignore rl_quantize.py --arch resnet50 --dataset imagenet --dataset_root data",
                 " --suffix ratio010 --preserve_ratio 0.1 --n_worker 120 --warmup -1 --train_episode ",
                 sobol_n,
                 " --data_bsize 128 --optimizer RS --val_size 10000 --train_size 20000",
                 sep = "")

    print(cmd)

    system(cmd)

    elapsed_time <- round((Sys.time() - start_time)[[1]] * 60)

    current_results <- read.csv("current_results.csv", header = TRUE)

    if(is.null(search_space)){
        search_space <- current_results
    } else{
        search_space <- bind_rows(search_space, current_results)
    }

    best_points <- filter(search_space, Top1 == min(Top1) | Top5 == min(Top5))
    best_points$id <- i
    best_points$elapsed <- elapsed_time
    best_points$points <- sobol_n

    if(is.null(results)){
        results <- best_points
    } else{
        results <- bind_rows(results, best_points)
    }

    write.csv(results,
              paste("rs_",
                    starting_sobol_n,
                    "_samples_",
                    i,
                    "_iterations.csv",
                    sep = ""),
              row.names = FALSE)
}
