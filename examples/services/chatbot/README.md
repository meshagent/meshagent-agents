# Sample Agent Service

This folder contains a sample Meshagent service which hosts a chatbot as a service in a room. To run the sample:

1. Build the docker image and push it to a docker repo.
2. Create a image pull secret if your docker repo does not allow public access
3. Create a service that uses this docker image and your image pull secret, add an environment variable MESHAGENT_PORT which matches the port that you register the service with.
4. Start a new session and the chatbot should automatically be added to the room when the room starts.

* note, if the room is already running when you create the service, the chatbot will not be added until the next time it starts.

To speed up image pulls we recomment using zstd images, to enable building of zstd images, run the following commands

```sh
docker buildx create --name zstd-builder --driver docker-container
docker buildx use zstd-builder
```

To build this container, use docker buildx to make a linux/amd64 image and push it to your docker repo. Replace YOUR_IMAGE_TAG with the tag you would like to use for the image.

```sh
docker buildx build . --platform linux/amd64 --output=type=image,name=YOUR_IMAGE_TAG,oci-mediatypes=true,compression=zstd,compression-level=3,force-compression=true,push=true
```

