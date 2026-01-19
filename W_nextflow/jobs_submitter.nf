//the following will be passed to the local processor
//------
params.input_folder = "/home/ulman/devel/NextFlow/inputs"
params.output_folder = "/home/ulman/devel/NextFlow/outputs"
params.processor = "/home/ulman/data/Kobe-Hackathon/seg_and_tra_pipeline/W_nextflow/processor.sh"
params.group_size = 1
// NB: Chunks the input folder files into groups, each of the size above,
//     one group is one tupple'd input -> if three files are required for
//     the processing, set group_size = 3
params.processor_needs_time = 1m
params.processor_needs_cpus = 2
//------

params.local_processor_command = "nextflow run /home/ulman/data/Kobe-Hackathon/seg_and_tra_pipeline/W_nextflow/folder_processor.nf"
params.local_processor_config_param = "-c /home/ulman/data/Kobe-Hackathon/seg_and_tra_pipeline/W_nextflow/config_node.nextflow"


process create_lists_of_files {
    maxForks params.get('max_pending_jobs', 10)

    // this statement is here only to associate SLURM settings with this process (task)
    // (would have no effect in the local config)
    clusterOptions params.get('cluster_options', '')

    // estimated time needed for this process (task) to finish
    // (would have no effect in the local config)
    time params.processor_needs_time
    cpus params.processor_needs_cpus

    input:
    path in_files_list

    script:
    """
    echo -n "JOB submitting ${in_files_list} in folder "
    pwd
    ${params.local_processor_command} ${params.local_processor_config_param} \
         --input_folder . \
         --output_folder ${params.output_folder} \
         --processor ${params.processor} \
         --group_size ${params.group_size}
    """
}


workflow {
    println("SUBMITTING JOBS...")
    println("CONSIDERING "+params.group_size+"-TUPLES...")

    file_list0 = files( "${params.input_folder}/*.tif" ).sort()

    max_jobs = params.get('max_pending_jobs', 10)
    cumulation_factor = (int)Math.ceil( file_list0.size() / (params.group_size*max_jobs) )
    println("CONSIDERING MAX "+cumulation_factor+" TASKS ON A NODE")

    file_list = channel.fromList( file_list0 )
    files_groups = file_list.buffer( size:params.group_size*cumulation_factor, remainder:true )

    create_lists_of_files( files_groups )
}
