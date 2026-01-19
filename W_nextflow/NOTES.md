## Secrets aka SSH access with per-user credentials

#### Prerequisities

The user is able to ssh-connect to his/her cluster
(where SLURM is available) using his/her private key.

#### Plan

On the server computer where BIOMERO is executed, install there Nextflow
and setup the following trio of secrets for each user:

```
nextflow secrets set OMEROUSER_SSH_HOST karolina.it4i.cz
nextflow secrets set OMEROUSER_SSH_LOGIN xulman
nextflow secrets set OMEROUSER_SSH_KEY_FILE path/to/local/xulman.ssh.priv.key
                                            (or maybe even the key itself?)
```

where `OMEROUSER` shall be substituted with the actual OMERO login
(or something else that identifies so that the "per-user" principle can
actually work).

If BIOMERO finds a way to implement its own "secrets store", the one from
Nextflow needs not be used...

In the end, BIOMERO could login on behalf of the user into a HPC place,
copy files in (with `scp`, ssh-based file transfer) and using there deployed
`pixi run --manifest-path somewhere/pixi.toml nextflow run ....` run a workflow,
a workflow such as the `jobs_submitter.nf` from here.
