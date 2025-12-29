2025-12-29 — Beginning Lesson 1.

I am beginning with a local-first approach as opposed to the cloud API. Specifically, I will be setting up vLLM for serving the model and running easy, web-UI-based tests, while also setting up a local download and lab via containerization and `docker-compose` on my Jetson Thor. This will be the primary box running the AI lessons unless we require a different, non-unified architecture for experiments, in which case we will transition to RTX GPUs. However, as it stands, I do not currently have the knowledge to differentiate between the pros and cons of either right now. We will begin by setting up containers on the Thor and a vLLM mode so we can work between the two.

We have added a `docker-compose.yml` to the repository to reliably and repeatably create environments for serving via vLLM and for running experiments in a lab-style environment.

In this workspace, any edits we make inside the container are saved back into our local files (via bind mounts).

To access JupyterLab for course work while hosting the model on Thor (and using Thor as the primary machine):

1) Start the lab container on Thor:

```bash
docker compose -f infra/compose.yml --env-file .env up lab
```

2) This builds/starts the Jupyter server. From the other machine (e.g., the Omni Sim box), create an SSH tunnel:

```bash
ssh -L 8888:127.0.0.1:8888 <ssh_user>@<thor_host>
```

3) Open JupyterLab locally in a browser:

```
http://127.0.0.1:8888/lab?token=<JUPYTER_TOKEN>
```

The shared cache is visible inside the container, which is good news because it means we can load the model weights and experiment as needed. For example:

```bash
echo $HF_HOME && ls -la /data/hf | head
```
