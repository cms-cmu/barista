apptainer exec \
    -B .:/srv \
    --nv \
    --pwd /srv \
    docker://chuyuanliu/heptools:ml \
    bash --init-file /entrypoint.sh
