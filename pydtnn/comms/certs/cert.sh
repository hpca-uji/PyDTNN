#!/usr/bin/env bash
# Generate self-signed localhost certicate

# Constants
self=$(realpath "$0")
impl="${self%/*}"

#echo 111111
#openssl req -new -x509 -days 3650 -extensions v3_ca -nodes -keyout $impl/ca.key -out $impl/ca.crt \
#    -subj "/CN=localhost/" \
#    -addext "subjectAltName=IP:127.0.0.1"

#echo 222222
#openssl req -newkey rsa -nodes -keyout $impl/server.key -out $impl/server.csr \
#    -subj "/CN=localhost/" \
#    -addext "subjectAltName=IP:127.0.0.1"

#echo 33333
#openssl x509 -req -in $impl/server.csr -CA $impl/ca.crt -CAkey $impl/ca.key -CAcreateserial -out $impl/server.crt -days 3650

#exit
# Compile
openssl req \
    -newkey rsa \
    -x509 -sha256 -nodes \
    -keyout "${impl:?}/key.pem" \
    -out "${impl:?}/cert.pem" \
    -days "3650" \
    -subj "/CN=localhost/" \
    -addext "subjectAltName=IP:127.0.0.1"