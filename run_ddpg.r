library(dplyr)

quiet <- function(x) {
  sink(tempfile())
  on.exit(sink())
  invisible(force(x))
}

iterations <- 10
results <- NULL

total_measurements <- 207

for(i in 1:iterations){
    search_space <- NULL
    start_time <- as.integer(format(Sys.time(), "%s"))

    cmd <- paste("python3 -W ignore rl_quantize.py --arch resnet50",
                 " --dataset imagenet --dataset_root data",
                 " --suffix ratio010 --preserve_ratio 0.1",
                 " --n_worker 120 --warmup 2 --train_episode ",
                 total_measurements,
                 " --use_top5",
                 " --data_bsize 128 --optimizer DDPG --val_size 10000",
                 " --train_size 20000",
                 sep = "")

    print(cmd)
    system(cmd)

    system("rm -r ../../save")

    elapsed_time <- as.integer(format(Sys.time(), "%s")) - start_time

    current_results <- read.csv("haq_results_log.csv", header = TRUE)

    if(is.null(search_space)){
        search_space <- current_results
    } else{
        search_space <- bind_rows(search_space, current_results) %>%
            distinct()
    }

    write.csv(search_space,
              paste("ddpg_",
                    total_measurements,
                    "_samples_",
                    i,
                    "_iteration_search_space.csv",
                    sep = ""),
              row.names = FALSE)

    best_points <- filter(search_space, Top5 == max(Top5))
    best_points$id <- i
    best_points$elapsed_seconds <- elapsed_time
    best_points$points <- total_measurements

    if(is.null(results)){
        results <- best_points
    } else{
        results <- bind_rows(results, best_points)
    }

    write.csv(results,
              paste("ddpg_",
                    total_measurements,
                    "_samples_",
                    iterations,
                    "_iterations.csv",
                    sep = ""),
              row.names = FALSE)
}
