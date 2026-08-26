"""Idempotent boto3 provisioning for the isolated ECS worker test stack.

No Terraform/CloudFormation: every resource is created, verified, and torn
down through explicit boto3 calls in ``infra.aws.orchestrator``, driven by
the ``infra.aws.cli`` command line.
"""
