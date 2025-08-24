#!/bin/env bash

cd downloaded && uv run hf upload niikun/duke-hf-demo demo_data.jsonl --repo-type dataset --private

