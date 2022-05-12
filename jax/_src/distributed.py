# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import functools

from typing import Optional

from absl import logging
from jax._src import cloud_tpu_init
from jax._src.lib import xla_bridge
from jax._src.lib import xla_client
from jax._src.lib import xla_extension

jax_service = None
distributed_client = None


def _find_tpu_coordinator_address():
  worker_endpoint = cloud_tpu_init.get_metadata('worker-network-endpoints')
  coordinator_addr = worker_endpoint.split(',')[0].split(':')[2]
  worker_id = cloud_tpu_init.get_metadata('agent-worker-number')
  current_process_addr = {w.split(':')[0][-1]: w.split(':')[2]
                          for w in worker_endpoint.split(',')}[worker_id]
  return coordinator_addr + ':8476', coordinator_addr == current_process_addr


def initialize(coordinator_address: Optional[str] = None,
               is_coordinator: Optional[bool] = None,
               num_processes: Optional[int] = None,
               process_id: Optional[int] = None):
  """Initialize distributed system for topology discovery.

  Currently, calling ``initialize`` sets up the multi-host GPU backend, and
  is not required for CPU or TPU backends.

  Args:
    coordinator_address: IP address and port of the coordinator. The choice of
      port does not matter, so long as the port is available on the coordinator
      and all processes agree on the port.
    is_coordinator: True if the current process is the coordinator. It defaults
      to process 0 as the coordinator.
    num_processes: Number of processes.
    process_id: Id of the current process.

  Raises:
    RuntimeError: `distributed.initialize` should only be called once.

  Example:

  Suppose there are two GPU hosts, and host 0 is the designated coordinator
  with address ``10.0.0.1:1234``. To initialize the GPU cluster, run the
  following commands before anything else.

  On host 0:

  >>> jax.distributed.initialize('10.0.0.1:1234', 2, 0)  # doctest: +SKIP

  On host 1:

  >>> jax.distributed.initialize('10.0.0.1:1234', 2, 1)  # doctest: +SKIP
  """

  coordinator_address = os.environ.get('JAX_COORDINATOR_ADDRESS',
                                       None) or coordinator_address
  if is_coordinator is None:
    is_coordinator = process_id == 0

  if cloud_tpu_init.running_in_cloud_tpu_vm:
    if coordinator_address is None:
      coordinator_address, is_coordinator = _find_tpu_coordinator_address()
    if num_processes is None:
      num_processes = xla_bridge.process_count()
    if process_id is None:
      process_id = xla_bridge.process_index()

  assert coordinator_address is not None
  assert is_coordinator is not None
  assert num_processes is not None
  assert process_id is not None

  if is_coordinator:
    global jax_service
    if jax_service is not None:
      raise RuntimeError('distributed.initialize should only be called once.')

    logging.info('Starting JAX distributed service on %s', coordinator_address)
    jax_service = xla_extension.get_distributed_runtime_service(
        coordinator_address, num_processes)

  global distributed_client
  if distributed_client is not None:
    raise RuntimeError('distributed.initialize should only be called once.')

  distributed_client = xla_extension.get_distributed_runtime_client(
      coordinator_address, process_id)
  logging.info('Connecting to JAX distributed service on %s', coordinator_address)
  distributed_client.connect()

  factory = functools.partial(
      xla_client.make_gpu_client,
      distributed_client,
      process_id,
      platform_name='cuda')
  xla_bridge.register_backend_factory('cuda', factory, priority=300)
  factory = functools.partial(
      xla_client.make_gpu_client,
      distributed_client,
      process_id,
      platform_name='rocm')
  xla_bridge.register_backend_factory('rocm', factory, priority=300)
