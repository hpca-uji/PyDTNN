#!/usr/bin/env bash
# Generate self-signed localhost certificate

# Constants
self=$(realpath "$0")
impl="${self%/*}"

# Compile
openssl req \
    -newkey rsa \
    -x509 -sha256 -nodes \
    -keyout "${impl:?}/key.pem" \
    -out "${impl:?}/cert.pem" \
    -days "3650" \
    -subj "/CN=localhost/" \
    -addext "subjectAltName=IP:127.0.0.1"