#!/bin/bash


source ./_env/project_config.bashrc
source ./_env/machine_config.bashrc
source ./_env/make_config.bashrc


if [[ $1 == shell_docker ]]; then
    ([ -z $DISPLAY ] && echo "no display" || xhost +local:docker) \
    && docker run \
        $OPTIONS_DOCKER_RUN \
        $MOUNTS_DOCKER \
        -it $DOCKER_USER/$DOCKER_NAME:latest \
            /bin/bash

elif [[ $1 == jupyterlab ]]; then
    ([ -z $DISPLAY ] && echo "no display" || xhost +local:docker) \
    && docker run \
        $OPTIONS_DOCKER_RUN \
        $MOUNTS_DOCKER \
        -it $DOCKER_USER/$DOCKER_NAME:latest \
            /bin/bash -c "
                source /root/.bashrc \
                && python3 -m jupyterlab --notebook-dir / --ip $JUPYTERLAB_HOST --port $JUPYTERLAB_PORT \
                    --allow-root --no-browser --ContentsManager.allow_hidden=True
            "

elif [[ $1 == shell_singularity ]]; then
    singularity exec \
        $OPTIONS_SINGULARITY_EXEC \
        $MOUNTS_SINGULARITY \
        ./_env/singularity.sif \
        /bin/bash

elif [[ $1 == docker_build ]]; then
   docker build -t $DOCKER_USER/$DOCKER_NAME:latest $PROJECT_DN/_env
elif [[ $1 == docker_push ]]; then
   docker push $DOCKER_USER/$DOCKER_NAME:latest
elif [[ $1 == docker_pull ]]; then
   docker pull $DOCKER_USER/$DOCKER_NAME:latest
elif [[ $1 == docker_stop ]]; then
   docker stop $(docker ps -aq) && docker rm $(docker ps -aq)

elif [[ $1 == singularity_build ]]; then
     -E singularity build \
        $PROJECT_DN/_env/singularity.sif \
        $PROJECT_DN/_env/singularity.def

fi


