#!/usr/bin/env python3
import random
import string

import aws_cdk as cdk

from aws_cdk import (
  Stack,
  aws_sagemaker
)
from constructs import Construct

from .dlc_image_urls import DLC_IMAGE_URL_BY_REGION

random.seed(47)

class EmbeddingEndpointStack(Stack):

  def __init__(self, scope: Construct, construct_id: str, sm_execution_role_arn, **kwargs) -> None:
    super().__init__(scope, construct_id, **kwargs)

    model_environment = {
      "MODEL_CACHE_ROOT": "/opt/ml/model",
      "SAGEMAKER_ENV": "1",
      "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600",
      "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
      "SAGEMAKER_PROGRAM": "inference.py",
      "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code/",
      "TS_DEFAULT_WORKERS_PER_MODEL": "1"
    }

    aws_region = kwargs['env'].region
    RANDOM_GUID = ''.join(random.sample(string.digits, k=7))
    embedding_model = aws_sagemaker.CfnModel(self, "EmbeddingModel",
      execution_role_arn=sm_execution_role_arn,
      model_name=f"gpt-j-6b-fp16-model-{RANDOM_GUID}",
      primary_container=aws_sagemaker.CfnModel.ContainerDefinitionProperty(
        environment=model_environment,
        #XXX: You need to checkout an available DLC(Deep Learning Container) image in your region.
        # For more information, see https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html
        image=f"{DLC_IMAGE_URL_BY_REGION[aws_region]}/pytorch-inference:1.12.0-gpu-py38",
        mode="SingleModel",
        model_data_url=f"s3://jumpstart-cache-prod-{cdk.Aws.REGION}/huggingface-infer/prepack/v1.0.0/infer-prepack-huggingface-textembedding-gpt-j-6b-fp16.tar.gz"
      )
    )

    embedding_endpoint_config = aws_sagemaker.CfnEndpointConfig(self, "EmbeddingEndpointConfig",
      production_variants=[aws_sagemaker.CfnEndpointConfig.ProductionVariantProperty(
        initial_variant_weight=1.0,
        model_name=embedding_model.model_name,
        variant_name=embedding_model.model_name,
        initial_instance_count=1,
        instance_type="ml.g5.2xlarge"
      )],
      endpoint_config_name=f"gpt-j-6b-fp16-endpoint-{RANDOM_GUID}"
    )
    embedding_endpoint_config.add_dependency(embedding_model)

    embedding_endpoint = aws_sagemaker.CfnEndpoint(self, "EmbeddingEndpoint",
      endpoint_config_name=embedding_endpoint_config.endpoint_config_name,
      endpoint_name=f"gpt-j-6b-fp16-endpoint-{RANDOM_GUID}"
    )
    embedding_endpoint.add_dependency(embedding_endpoint_config)

    cdk.CfnOutput(self, 'EmbeddingEndpointName', value=embedding_endpoint.endpoint_name)
