# Artifact

Experiment outputs (hydra run directories, tensorboard logs, trained model checkpoints, test-time
rollout records, plots) are written here by every launcher, under

```
artifact/<algorithm.name>/<experiment>/<date>/<time>/...
```

The directory is VCS ignored and bind mounted read-and-write in the docker container
(see `docker/docker-compose.yaml`).
