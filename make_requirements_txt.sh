#!/bin/bash
poetry export --only main -f requirements.txt --output requirements.txt --without-hashes
