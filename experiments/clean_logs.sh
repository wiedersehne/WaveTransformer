#!/bin/sh

# Clean the log files. Just a quick script to save me some seconds :)

rm -r logs/checkpoints
rm -r logs/lightning_logs
rm -r logs/embeddings/tb/*

#rm logs/embeddings/imgs/*
#rm logs/embeddings/embs/*
