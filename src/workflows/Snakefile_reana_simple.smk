analysis_container = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest"
output_dir = "reana_output"

rule all:
    input:
        f"{output_dir}/hello.txt"

rule hello_world:
    container: analysis_container
    output:
        f"{output_dir}/hello.txt"
    shell:
        """
        mkdir -p {output_dir}
        echo "Hello, World! Running on REANA." > {output}
        echo "Hostname: $(hostname)" >> {output}
        echo "Time: $(date)" >> {output}
        """
