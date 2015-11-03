FROM andrewosh/binder-base

MAINTAINER Davis Bennett <davis.v.bennett@gmail.com> 

USER root

RUN apt-get update
RUN apt-get -y --force-yes install ffmpeg libavcodec-extra-52 libavdevice-extra-52 libavfilter-extra-0 libavformat-extra-52 libavutil-extra-49 libpostproc-extra-51 libswscale-extra-0

USER main
