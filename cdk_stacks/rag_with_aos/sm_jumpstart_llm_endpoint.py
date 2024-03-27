#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# vim: tabstop=2 shiftwidth=2 softtabstop=2 expandtab

import random
import string

import aws_cdk as cdk

from aws_cdk import (
  Stack
)
from constructs import Construct

from cdklabs.generative_ai_cdk_constructs import (
  JumpStartSageMakerEndpoint,
  JumpStartModel,
  SageMakerInstanceType
)

random.seed(47)


class SageMakerJumpStartLLMEndpointStack(Stack):

  def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
    super().__init__(scope, construct_id, **kwargs)

    jumpstart_model_id = self.node.try_get_context('jumpstart_llm_model_id') or 'META_TEXTGENERATION_LLAMA_2_7B_2_1_0'
    llm_endpoint_name = self.node.try_get_context('llm_endpoint_name') or 'llama2-7b'

    RANDOM_GUID = ''.join(random.sample(string.digits, k=7))
    endpoint_name = f"{llm_endpoint_name}-{RANDOM_GUID}"

    #XXX: Available JumStart Model List
    # https://github.com/awslabs/generative-ai-cdk-constructs/blob/main/src/patterns/gen-ai/aws-model-deployment-sagemaker/jumpstart-model.ts
    llm_endpoint = JumpStartSageMakerEndpoint(self, 'LLMEndpoint',
      model=JumpStartModel.of(jumpstart_model_id.upper()),
      accept_eula=True,
      instance_type=SageMakerInstanceType.ML_G5_2_XLARGE,
      endpoint_name=endpoint_name
    )

    cdk.CfnOutput(self, 'LLMEndpointName', value=llm_endpoint.cfn_endpoint.endpoint_name,
      export_name=f'{self.stack_name}-LLMEndpointName')
    cdk.CfnOutput(self, 'LLMEndpointArn', value=llm_endpoint.endpoint_arn,
      export_name=f'{self.stack_name}-LLMEndpointArn')