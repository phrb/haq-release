library(randtoolbox)
library(dplyr)
library(rsm)
library(DiceKriging)
library(DiceOptim)
library(future.apply)

plan(multiprocess, workers = 256)

quiet <- function(x) {
  sink(tempfile())
  on.exit(sink())
  invisible(force(x))
}

iterations <- 10

results <- NULL

sobol_dim <- 54 * 2
starting_sobol_n <- (1 * sobol_dim) + 1
sobol_n <- starting_sobol_n

bit_min <- 1
bit_max <- 8
perturbation_range <- 1 * (bit_min / bit_max)

gpr_iterations <- 20
gpr_added_points <- 3

gpr_added_neighbours <- 2
gpr_neighbourhood_factor <- 1000

gpr_sample_size <- 200 * sobol_dim

total_measurements <- starting_sobol_n + (gpr_iterations * (gpr_added_points + gpr_added_neighbours))

for(i in 1:iterations){
    gpr_sample <- NULL
    search_space <- NULL

    temp_sobol <- sobol(n = sobol_n,
                        dim = sobol_dim,
                        scrambling = 1,
                        seed = as.integer((99999 - 10000) * runif(1) + 10000),
                        init = TRUE)

    rm(temp_sobol)
    quiet(gc())

    design <- sobol(n = sobol_n,
                    dim = sobol_dim,
                    scrambling = 1,
                    seed = as.integer((99999 - 10000) * runif(1) + 10000),
                    init = FALSE)

    df_design <- data.frame(design)

    names(df_design) <- c(rbind(paste("W",
                                      seq(1:(sobol_dim / 2)),
                                      sep = ""),
                                paste("A",
                                      seq(1:(sobol_dim / 2)),
                                      sep = "")))

    write.csv(df_design, "current_design.csv", row.names = FALSE)

    start_time <- as.integer(format(Sys.time(), "%s"))

    cmd <- paste("python3 -W ignore rl_quantize.py --arch resnet50",
                 " --dataset imagenet --dataset_root data",
                 " --suffix ratio010 --preserve_ratio 0.1",
                 " --n_worker 120 --warmup -1 --train_episode ",
                 sobol_n,
                 " --use_top5",
                 " --data_bsize 128 --optimizer RS --val_size 10000",
                 " --train_size 20000",
                 sep = "")

    print(cmd)
    system(cmd)

    system("rm -r ../../save")

    current_results <- read.csv("current_results.csv", header = TRUE)

    if(is.null(search_space)){
        search_space <- current_results
    } else{
        search_space <- bind_rows(search_space, current_results) %>%
            distinct()
    }

    write.csv(search_space,
              paste("gpr_",
                    total_measurements,
                    "_samples_",
                    i,
                    "_iteration_search_space.csv",
                    sep = ""),
              row.names = FALSE)

    for(j in 1:gpr_iterations){
        # Optimzing for Top5
        print("Starting reg")
        gpr_model <- km(design = select(search_space, -Top5, -Top1),
                        response = ((rowSums(select(search_space, -Top5, -Top1)) / sobol_dim) +
                                    ((100.0 - search_space$Top5) / 100.0)) / 2,
                        control = list(pop.size = 400,
                                       BFGSburnin = 500))

        print("Generating Sample")
        new_sample <- sobol(n = gpr_sample_size,
                            dim = sobol_dim,
                            scrambling = 1,
                            seed = as.integer((99999 - 10000) * runif(1) + 10000),
                            init = FALSE)

        new_sample <- data.frame(new_sample)

        names(new_sample) <- c(rbind(paste("W",
                                           seq(1:(sobol_dim / 2)),
                                           sep = ""),
                                     paste("A",
                                           seq(1:(sobol_dim / 2)),
                                           sep = "")))

        if(is.null(gpr_sample)){
            gpr_sample <- new_sample
        } else{
            gpr_sample <- bind_rows(gpr_sample, new_sample) %>%
                distinct()
        }

        print("Computing EI")
        gpr_sample$expected_improvement <- future_apply(gpr_sample,
                                                        1,
                                                        EI,
                                                        gpr_model)

        gpr_selected_points <- gpr_sample %>%
            arrange(desc(expected_improvement))

        gpr_sample <- select(gpr_sample, -expected_improvement)

        gpr_selected_points <- select(gpr_selected_points[1:gpr_added_points, ],
                                      -expected_improvement)

        print("Generating perturbation sample")
        perturbation <- sobol(n = gpr_added_points * gpr_neighbourhood_factor,
                              dim = sobol_dim,
                              scrambling = 1,
                              seed = as.integer((99999 - 10000) * runif(1) + 10000),
                              init = FALSE)

        perturbation <- data.frame(perturbation)

        names(perturbation) <- c(rbind(paste("W",
                                            seq(1:(sobol_dim / 2)),
                                            sep = ""),
                                      paste("A",
                                            seq(1:(sobol_dim / 2)),
                                            sep = "")))

        perturbation <- (2 * perturbation_range * perturbation) - perturbation_range

        gpr_selected_neighbourhood <- gpr_selected_points %>%
            slice(rep(row_number(), gpr_neighbourhood_factor))

        gpr_selected_neighbourhood <- gpr_selected_neighbourhood + perturbation

        gpr_selected_neighbourhood[gpr_selected_neighbourhood < 0.0] <- 0.124
        gpr_selected_neighbourhood[gpr_selected_neighbourhood > 1.0] <- 0.876

        gpr_sample <- bind_rows(gpr_sample, gpr_selected_neighbourhood) %>%
            distinct()

        gpr_selected_points <- bind_rows(gpr_selected_points,
                                         gpr_selected_neighbourhood)
        gpr_selected_points <- gpr_selected_points %>%
            distinct()

        print("Computing perturbed EI")
        gpr_selected_points$expected_improvement <- future_apply(gpr_selected_points,
                                                                 1,
                                                                 EI,
                                                                 gpr_model)

        gpr_selected_points <- gpr_selected_points %>%
            arrange(desc(expected_improvement))

        gpr_selected_points <- select(gpr_selected_points[1:(gpr_added_points +
                                                             gpr_added_neighbours), ],
                                      -expected_improvement)

        df_design <- data.frame(gpr_selected_points)
        names(df_design) <- c(rbind(paste("W",
                                          seq(1:(sobol_dim / 2)),
                                          sep = ""),
                                    paste("A",
                                          seq(1:(sobol_dim / 2)),
                                          sep = "")))

        write.csv(df_design, "current_design.csv", row.names = FALSE)

        cmd <- paste("python3 -W ignore rl_quantize.py --arch resnet50",
                     " --dataset imagenet --dataset_root data",
                     " --suffix ratio010 --preserve_ratio 0.1",
                     " --n_worker 120 --warmup -1 --train_episode ",
                     gpr_added_points + gpr_added_neighbours,
                     " --use_top5",
                     " --data_bsize 128 --optimizer RS --val_size 10000",
                     " --train_size 20000",
                     sep = "")

        print(cmd)
        system(cmd)

        system("rm -r ../../save")

        current_results <- read.csv("current_results.csv", header = TRUE)

        if(is.null(search_space)){
            search_space <- current_results
        } else{
            search_space <- bind_rows(search_space, current_results) %>%
                distinct()
        }

        write.csv(search_space,
                  paste("gpr_",
                        total_measurements,
                        "_samples_",
                        i,
                        "_iteration_search_space.csv",
                        sep = ""),
                  row.names = FALSE)
    }

    elapsed_time <- as.integer(format(Sys.time(), "%s")) - start_time

    # Optimizing for Top5 (Top1, Model Size, Latency... ?)
    best_points <- filter(search_space, Top5 == max(Top5))

    best_points$id <- i
    best_points$elapsed_seconds <- elapsed_time
    best_points$points <- total_measurements

    best_points$gpr_iterations <- gpr_iterations
    best_points$gpr_added_points <- gpr_added_points
    best_points$perturbation_range <- perturbation_range
    best_points$gpr_neighbourhood <- gpr_neighbourhood_factor
    best_points$gpr_sample_size <- gpr_sample_size

    if(is.null(results)){
        results <- best_points
    } else{
        results <- bind_rows(results, best_points)
    }

    write.csv(results,
              paste("gpr_s_only",
                    total_measurements,
                    "_samples_",
                    iterations,
                    "_iterations.csv",
                    sep = ""),
              row.names = FALSE)
}
