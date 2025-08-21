"""DigitalOcean infrastructure provisioning for VSR API."""

import os
import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import digitalocean
import requests

from vsr_shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DropletConfig:
    """Configuration for DigitalOcean droplet."""
    name: str
    size: str
    image: str
    region: str
    tags: List[str]
    ssh_keys: List[str]
    user_data: Optional[str] = None
    monitoring: bool = True
    backups: bool = False


@dataclass
class InfrastructureConfig:
    """Configuration for complete infrastructure setup."""
    project_name: str
    region: str
    ssh_key_name: str
    vpc_name: str
    api_droplet: DropletConfig
    gpu_droplet: DropletConfig
    firewall_rules: Dict[str, Any]


class DigitalOceanProvisioner:
    """DigitalOcean infrastructure provisioner."""

    def __init__(self, api_token: str):
        """
        Initialize DO provisioner.

        Args:
            api_token: DigitalOcean API token
        """
        self.api_token = api_token
        self.manager = digitalocean.Manager(token=api_token)
        self.resources = {
            "droplets": {},
            "vpc": None,
            "firewall": None,
            "ssh_key": None,
        }

    async def provision_infrastructure(self, config: InfrastructureConfig) -> Dict[str, Any]:
        """
        Provision complete infrastructure.

        Args:
            config: Infrastructure configuration

        Returns:
            Dict containing provisioned resources
        """
        try:
            logger.info(f"Starting infrastructure provisioning for {config.project_name}")

            # Step 1: Create or get SSH key
            ssh_key = await self._setup_ssh_key(config.ssh_key_name)
            self.resources["ssh_key"] = ssh_key

            # Step 2: Create VPC for private networking
            vpc = await self._create_vpc(config.vpc_name, config.region)
            self.resources["vpc"] = vpc

            # Step 3: Create API droplet
            api_droplet = await self._create_droplet(
                config.api_droplet,
                vpc_uuid=vpc.id,
                ssh_key_ids=[ssh_key.id]
            )
            self.resources["droplets"]["api"] = api_droplet

            # Step 4: Create GPU droplet
            gpu_droplet = await self._create_droplet(
                config.gpu_droplet,
                vpc_uuid=vpc.id,
                ssh_key_ids=[ssh_key.id]
            )
            self.resources["droplets"]["gpu"] = gpu_droplet

            # Step 5: Wait for droplets to be ready
            await self._wait_for_droplets([api_droplet, gpu_droplet])

            # Step 6: Create firewall rules
            firewall = await self._create_firewall(
                f"{config.project_name}-firewall",
                config.firewall_rules,
                [api_droplet.id, gpu_droplet.id]
            )
            self.resources["firewall"] = firewall

            # Step 7: Configure private networking
            await self._configure_private_networking([api_droplet, gpu_droplet])

            logger.info("Infrastructure provisioning completed successfully")

            return {
                "status": "success",
                "resources": {
                    "api_droplet": {
                        "id": api_droplet.id,
                        "name": api_droplet.name,
                        "ip_address": api_droplet.ip_address,
                        "private_ip_address": api_droplet.private_ip_address,
                        "status": api_droplet.status,
                    },
                    "gpu_droplet": {
                        "id": gpu_droplet.id,
                        "name": gpu_droplet.name,
                        "ip_address": gpu_droplet.ip_address,
                        "private_ip_address": gpu_droplet.private_ip_address,
                        "status": gpu_droplet.status,
                    },
                    "vpc": {
                        "id": vpc.id,
                        "name": vpc.name,
                        "ip_range": vpc.ip_range,
                    },
                    "firewall": {
                        "id": firewall.id,
                        "name": firewall.name,
                    },
                    "ssh_key": {
                        "id": ssh_key.id,
                        "name": ssh_key.name,
                        "fingerprint": ssh_key.fingerprint,
                    },
                }
            }

        except Exception as e:
            logger.error(f"Infrastructure provisioning failed: {e}")
            await self._cleanup_resources()
            raise

    async def _setup_ssh_key(self, key_name: str) -> digitalocean.SSHKey:
        """Setup SSH key for droplet access."""
        try:
            # Check if key already exists
            ssh_keys = self.manager.get_all_sshkeys()
            for key in ssh_keys:
                if key.name == key_name:
                    logger.info(f"Using existing SSH key: {key_name}")
                    return key

            # Create new SSH key if not exists
            ssh_public_key = os.getenv("SSH_PUBLIC_KEY")
            if not ssh_public_key:
                raise ValueError("SSH_PUBLIC_KEY environment variable not set")

            ssh_key = digitalocean.SSHKey(
                token=self.api_token,
                name=key_name,
                public_key=ssh_public_key
            )
            ssh_key.create()

            logger.info(f"Created SSH key: {key_name}")
            return ssh_key

        except Exception as e:
            logger.error(f"Failed to setup SSH key: {e}")
            raise

    async def _create_vpc(self, vpc_name: str, region: str) -> digitalocean.VPC:
        """Create VPC for private networking."""
        try:
            # Check if VPC already exists
            vpcs = self.manager.get_all_vpcs()
            for vpc in vpcs:
                if vpc.name == vpc_name and vpc.region == region:
                    logger.info(f"Using existing VPC: {vpc_name}")
                    return vpc

            # Create new VPC
            vpc = digitalocean.VPC(
                token=self.api_token,
                name=vpc_name,
                region=region,
                ip_range="10.0.0.0/16"
            )
            vpc.create()

            logger.info(f"Created VPC: {vpc_name} in {region}")
            return vpc

        except Exception as e:
            logger.error(f"Failed to create VPC: {e}")
            raise

    async def _create_droplet(
        self,
        config: DropletConfig,
        vpc_uuid: str,
        ssh_key_ids: List[int]
    ) -> digitalocean.Droplet:
        """Create a droplet with specified configuration."""
        try:
            logger.info(f"Creating droplet: {config.name}")

            droplet = digitalocean.Droplet(
                token=self.api_token,
                name=config.name,
                region=config.region,
                image=config.image,
                size_slug=config.size,
                ssh_keys=ssh_key_ids,
                tags=config.tags,
                monitoring=config.monitoring,
                backups=config.backups,
                vpc_uuid=vpc_uuid,
                user_data=config.user_data
            )

            droplet.create()
            logger.info(f"Droplet {config.name} creation initiated")

            return droplet

        except Exception as e:
            logger.error(f"Failed to create droplet {config.name}: {e}")
            raise

    async def _wait_for_droplets(
        self,
        droplets: List[digitalocean.Droplet],
        timeout: int = 300
    ) -> None:
        """Wait for droplets to be ready."""
        logger.info("Waiting for droplets to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            all_ready = True
            
            for droplet in droplets:
                droplet.load()
                if droplet.status != "active":
                    all_ready = False
                    logger.info(f"Droplet {droplet.name} status: {droplet.status}")
                    break

            if all_ready:
                logger.info("All droplets are ready")
                return

            await asyncio.sleep(10)

        raise TimeoutError("Timeout waiting for droplets to be ready")

    async def _create_firewall(
        self,
        firewall_name: str,
        rules: Dict[str, Any],
        droplet_ids: List[int]
    ) -> digitalocean.Firewall:
        """Create firewall with security rules."""
        try:
            logger.info(f"Creating firewall: {firewall_name}")

            # Define inbound rules
            inbound_rules = [
                # SSH access
                {
                    "protocol": "tcp",
                    "ports": "22",
                    "sources": {
                        "addresses": ["0.0.0.0/0", "::/0"]
                    }
                },
                # HTTP/HTTPS for API
                {
                    "protocol": "tcp",
                    "ports": "80",
                    "sources": {
                        "addresses": ["0.0.0.0/0", "::/0"]
                    }
                },
                {
                    "protocol": "tcp",
                    "ports": "443",
                    "sources": {
                        "addresses": ["0.0.0.0/0", "::/0"]
                    }
                },
                # Private network communication
                {
                    "protocol": "tcp",
                    "ports": "1-65535",
                    "sources": {
                        "addresses": ["10.0.0.0/16"]
                    }
                }
            ]

            # Add custom rules from config
            if "inbound" in rules:
                inbound_rules.extend(rules["inbound"])

            # Define outbound rules (allow all by default)
            outbound_rules = [
                {
                    "protocol": "tcp",
                    "ports": "1-65535",
                    "destinations": {
                        "addresses": ["0.0.0.0/0", "::/0"]
                    }
                },
                {
                    "protocol": "udp",
                    "ports": "1-65535",
                    "destinations": {
                        "addresses": ["0.0.0.0/0", "::/0"]
                    }
                }
            ]

            firewall = digitalocean.Firewall(
                token=self.api_token,
                name=firewall_name,
                inbound_rules=inbound_rules,
                outbound_rules=outbound_rules,
                droplet_ids=droplet_ids
            )

            firewall.create()
            logger.info(f"Firewall {firewall_name} created successfully")

            return firewall

        except Exception as e:
            logger.error(f"Failed to create firewall: {e}")
            raise

    async def _configure_private_networking(
        self,
        droplets: List[digitalocean.Droplet]
    ) -> None:
        """Configure private networking between droplets."""
        try:
            logger.info("Configuring private networking...")

            # Reload droplets to get latest network info
            for droplet in droplets:
                droplet.load()

            # Log private IP addresses for reference
            for droplet in droplets:
                if droplet.private_ip_address:
                    logger.info(f"{droplet.name} private IP: {droplet.private_ip_address}")
                else:
                    logger.warning(f"{droplet.name} has no private IP address")

            logger.info("Private networking configuration completed")

        except Exception as e:
            logger.error(f"Failed to configure private networking: {e}")
            raise

    async def _cleanup_resources(self) -> None:
        """Cleanup created resources on failure."""
        logger.info("Cleaning up resources...")

        try:
            # Destroy droplets
            for droplet_type, droplet in self.resources["droplets"].items():
                if droplet:
                    logger.info(f"Destroying {droplet_type} droplet: {droplet.name}")
                    droplet.destroy()

            # Destroy firewall
            if self.resources["firewall"]:
                logger.info(f"Destroying firewall: {self.resources['firewall'].name}")
                self.resources["firewall"].destroy()

            # Destroy VPC (only if empty)
            if self.resources["vpc"]:
                logger.info(f"Destroying VPC: {self.resources['vpc'].name}")
                self.resources["vpc"].destroy()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Check health of provisioned infrastructure."""
        try:
            health_status = {
                "healthy": True,
                "resources": {},
                "errors": []
            }

            # Check droplets
            for droplet_type, droplet in self.resources["droplets"].items():
                if droplet:
                    droplet.load()
                    health_status["resources"][droplet_type] = {
                        "status": droplet.status,
                        "ip_address": droplet.ip_address,
                        "private_ip": droplet.private_ip_address,
                    }
                    
                    if droplet.status != "active":
                        health_status["healthy"] = False
                        health_status["errors"].append(f"{droplet_type} droplet not active")

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "errors": [f"Health check failed: {e}"]
            }


def create_default_config(project_name: str, region: str = "nyc3") -> InfrastructureConfig:
    """Create default infrastructure configuration."""
    
    # User data script for initial setup
    user_data_script = """#!/bin/bash
# Update system
apt-get update
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker $USER

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
mkdir -p /opt/vsr-api
chown $USER:$USER /opt/vsr-api

# Install monitoring tools
apt-get install -y htop iotop nethogs
"""

    gpu_user_data_script = user_data_script + """
# Install NVIDIA drivers and Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-docker2
systemctl restart docker

# Verify NVIDIA setup
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
"""

    return InfrastructureConfig(
        project_name=project_name,
        region=region,
        ssh_key_name=f"{project_name}-ssh-key",
        vpc_name=f"{project_name}-vpc",
        api_droplet=DropletConfig(
            name=f"{project_name}-api",
            size="s-2vcpu-4gb",  # 2 vCPU, 4GB RAM
            image="ubuntu-22-04-x64",
            region=region,
            tags=["api", "production", project_name],
            ssh_keys=[],
            user_data=user_data_script,
            monitoring=True,
            backups=True,
        ),
        gpu_droplet=DropletConfig(
            name=f"{project_name}-gpu-worker",
            size="g-2vcpu-8gb-nvidia-rtx-4000",  # GPU droplet
            image="ubuntu-22-04-x64",
            region=region,
            tags=["gpu", "worker", "production", project_name],
            ssh_keys=[],
            user_data=gpu_user_data_script,
            monitoring=True,
            backups=False,  # GPU workers can be ephemeral
        ),
        firewall_rules={
            "inbound": [
                # Custom API port if needed
                {
                    "protocol": "tcp",
                    "ports": "8000",
                    "sources": {
                        "addresses": ["0.0.0.0/0", "::/0"]
                    }
                }
            ]
        }
    )


async def provision_vsr_infrastructure(
    api_token: str,
    project_name: str = "vsr-api",
    region: str = "nyc3"
) -> Dict[str, Any]:
    """
    Provision complete VSR infrastructure on DigitalOcean.

    Args:
        api_token: DigitalOcean API token
        project_name: Project name for resource naming
        region: DigitalOcean region

    Returns:
        Provisioning results
    """
    provisioner = DigitalOceanProvisioner(api_token)
    config = create_default_config(project_name, region)
    
    return await provisioner.provision_infrastructure(config)
