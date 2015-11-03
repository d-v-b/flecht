FROM andrewosh/binder-base

MAINTAINER Davis Bennett <davis.v.bennett@gmail.com> 

USER root

RUN apt-get update
RUN  apt-get install -y yasm nasm \
                build-essential automake autoconf \
                libtool pkg-config libcurl4-openssl-dev \
                intltool libxml2-dev libgtk2.0-dev \
                libnotify-dev libglib2.0-dev libevent-dev \
                checkinstall git x264
                
USER main
RUN git clone git://git.videolan.org/ffmpeg.git && cd ffmpeg &&\
    ./configure --enable-libx264 --enable-gpl &&\
    make -j 8 && make install 
