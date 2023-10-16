#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import uuid
import sys

from torch.distributed.run import parse_args, config_from_args
from torch.distributed.elastic.utils.logging import get_logger

from megatron.elastic.agent import ElasticAgent
from megatron.elastic.agent import elastic_launch

logger = get_logger(__name__)

def run(args):
    if args.standalone:
        args.rdzv_backend = "c10d"
        args.rdzv_endpoint = "localhost:0"
        args.rdzv_id = str(uuid.uuid4())
        logger.info(
            "\n**************************************\n"
            "Rendezvous info:\n"
            "--rdzv-backend=%s "
            "--rdzv-endpoint=%s "
            "--rdzv-id=%s\n"
            "**************************************\n",
            args.rdzv_backend, args.rdzv_endpoint, args.rdzv_id
        )

    config, cmd, cmd_args = config_from_args(args)
    elastic_launch(
        config=config,
        entrypoint=cmd,
    )(*cmd_args)

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    run(args)

if __name__ == '__main__':
    main()
