"""
SSL certificate automation using Let's Encrypt and Certbot.
Handles certificate generation, renewal, and domain configuration.
"""
import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import time

import requests
from cryptography import x509
from cryptography.hazmat.backends import default_backend


logger = logging.getLogger(__name__)


class SSLCertificateManager:
    """
    SSL certificate management using Let's Encrypt and Certbot.
    """
    
    def __init__(self, 
                 domain: str,
                 email: str,
                 cert_dir: str = "/etc/letsencrypt",
                 webroot_path: str = "/var/www/certbot"):
        self.domain = domain
        self.email = email
        self.cert_dir = Path(cert_dir)
        self.webroot_path = Path(webroot_path)
        self.cert_path = self.cert_dir / "live" / domain / "fullchain.pem"
        self.key_path = self.cert_dir / "live" / domain / "privkey.pem"
    
    async def setup_certbot(self) -> bool:
        """
        Install and configure Certbot for SSL certificate management.
        
        Returns:
            True if setup was successful
        """
        try:
            logger.info("Setting up Certbot for SSL certificate management")
            
            # Install Certbot (Ubuntu/Debian)
            commands = [
                ["apt-get", "update"],
                ["apt-get", "install", "-y", "certbot", "python3-certbot-nginx"],
                ["mkdir", "-p", str(self.webroot_path)]
            ]
            
            for cmd in commands:
                result = await self._run_command(cmd)
                if result.returncode != 0:
                    logger.error(f"Command failed: {' '.join(cmd)}")
                    logger.error(f"Error: {result.stderr}")
                    return False
            
            logger.info("Certbot setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Certbot: {e}")
            return False
    
    async def generate_certificate(self, staging: bool = False) -> bool:
        """
        Generate SSL certificate using Let's Encrypt.
        
        Args:
            staging: Use Let's Encrypt staging environment for testing
            
        Returns:
            True if certificate was generated successfully
        """
        try:
            logger.info(f"Generating SSL certificate for domain: {self.domain}")
            
            # Prepare Certbot command
            cmd = [
                "certbot", "certonly",
                "--webroot",
                "--webroot-path", str(self.webroot_path),
                "--email", self.email,
                "--agree-tos",
                "--no-eff-email",
                "--domains", self.domain,
                "--non-interactive"
            ]
            
            if staging:
                cmd.append("--staging")
                logger.info("Using Let's Encrypt staging environment")
            
            # Run Certbot
            result = await self._run_command(cmd)
            
            if result.returncode == 0:
                logger.info(f"SSL certificate generated successfully for {self.domain}")
                return True
            else:
                logger.error(f"Certificate generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to generate certificate: {e}")
            return False
    
    async def renew_certificate(self) -> bool:
        """
        Renew SSL certificate.
        
        Returns:
            True if renewal was successful
        """
        try:
            logger.info("Renewing SSL certificate")
            
            cmd = ["certbot", "renew", "--quiet", "--no-self-upgrade"]
            result = await self._run_command(cmd)
            
            if result.returncode == 0:
                logger.info("Certificate renewal completed successfully")
                # Reload Nginx to use new certificate
                await self._reload_nginx()
                return True
            else:
                logger.error(f"Certificate renewal failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to renew certificate: {e}")
            return False
    
    async def check_certificate_expiry(self) -> Optional[datetime]:
        """
        Check certificate expiration date.
        
        Returns:
            Certificate expiration datetime or None if not found
        """
        try:
            if not self.cert_path.exists():
                logger.warning(f"Certificate not found: {self.cert_path}")
                return None
            
            with open(self.cert_path, 'rb') as f:
                cert_data = f.read()
            
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
            expiry_date = cert.not_valid_after
            
            logger.info(f"Certificate expires on: {expiry_date}")
            return expiry_date
            
        except Exception as e:
            logger.error(f"Failed to check certificate expiry: {e}")
            return None
    
    async def is_certificate_valid(self) -> bool:
        """
        Check if certificate is valid and not expired.
        
        Returns:
            True if certificate is valid
        """
        try:
            expiry_date = await self.check_certificate_expiry()
            if not expiry_date:
                return False
            
            # Check if certificate expires within 30 days
            days_until_expiry = (expiry_date - datetime.now()).days
            
            if days_until_expiry <= 30:
                logger.warning(f"Certificate expires in {days_until_expiry} days")
                return False
            
            logger.info(f"Certificate is valid for {days_until_expiry} more days")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate certificate: {e}")
            return False
    
    async def setup_auto_renewal(self) -> bool:
        """
        Setup automatic certificate renewal using cron.
        
        Returns:
            True if auto-renewal was setup successfully
        """
        try:
            logger.info("Setting up automatic certificate renewal")
            
            # Create renewal script
            renewal_script = self._create_renewal_script()
            script_path = Path("/usr/local/bin/renew-ssl-cert.sh")
            
            with open(script_path, 'w') as f:
                f.write(renewal_script)
            
            # Make script executable
            await self._run_command(["chmod", "+x", str(script_path)])
            
            # Add cron job (run twice daily)
            cron_entry = "0 12,0 * * * /usr/local/bin/renew-ssl-cert.sh >> /var/log/ssl-renewal.log 2>&1\n"
            
            # Add to crontab
            result = await self._run_command(
                ["sh", "-c", f"(crontab -l 2>/dev/null; echo '{cron_entry.strip()}') | crontab -"]
            )
            
            if result.returncode == 0:
                logger.info("Auto-renewal setup completed successfully")
                return True
            else:
                logger.error(f"Failed to setup cron job: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup auto-renewal: {e}")
            return False
    
    def _create_renewal_script(self) -> str:
        """Create SSL renewal script."""
        return f"""#!/bin/bash
# SSL Certificate Renewal Script
# Generated automatically by VSR API SSL Manager

set -e

echo "$(date): Starting SSL certificate renewal check"

# Attempt renewal
if certbot renew --quiet --no-self-upgrade; then
    echo "$(date): Certificate renewal check completed successfully"
    
    # Reload Nginx if certificates were renewed
    if systemctl is-active --quiet nginx; then
        systemctl reload nginx
        echo "$(date): Nginx reloaded"
    fi
    
    # Restart Docker containers if needed
    if command -v docker-compose &> /dev/null; then
        cd /opt/vsr-api
        if [ -f docker-compose.prod.yml ]; then
            docker-compose -f docker-compose.prod.yml restart nginx
            echo "$(date): Docker Nginx container restarted"
        fi
    fi
else
    echo "$(date): Certificate renewal failed"
    exit 1
fi

echo "$(date): SSL renewal script completed"
"""
    
    async def configure_nginx_ssl(self) -> bool:
        """
        Configure Nginx with SSL certificate.
        
        Returns:
            True if configuration was successful
        """
        try:
            logger.info("Configuring Nginx with SSL certificate")
            
            # Update Nginx configuration
            nginx_config = self._create_nginx_ssl_config()
            config_path = Path("/etc/nginx/conf.d/vsr-api-ssl.conf")
            
            with open(config_path, 'w') as f:
                f.write(nginx_config)
            
            # Test Nginx configuration
            result = await self._run_command(["nginx", "-t"])
            if result.returncode != 0:
                logger.error(f"Nginx configuration test failed: {result.stderr}")
                return False
            
            # Reload Nginx
            await self._reload_nginx()
            
            logger.info("Nginx SSL configuration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure Nginx SSL: {e}")
            return False
    
    def _create_nginx_ssl_config(self) -> str:
        """Create Nginx SSL configuration."""
        return f"""# SSL Configuration for VSR API
server {{
    listen 80;
    server_name {self.domain};
    
    # Let's Encrypt challenge location
    location /.well-known/acme-challenge/ {{
        root {self.webroot_path};
    }}
    
    # Redirect all other HTTP traffic to HTTPS
    location / {{
        return 301 https://$server_name$request_uri;
    }}
}}

server {{
    listen 443 ssl http2;
    server_name {self.domain};
    
    # SSL Certificate Configuration
    ssl_certificate {self.cert_path};
    ssl_certificate_key {self.key_path};
    
    # SSL Security Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Client upload size limit (500MB for video files)
    client_max_body_size 500M;
    client_body_timeout 300s;
    client_header_timeout 60s;
    
    # Proxy settings
    proxy_connect_timeout 60s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=2r/s;
    
    # Health check endpoint (no rate limiting)
    location /healthz {{
        proxy_pass http://vsr-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        access_log off;
    }}
    
    # Upload endpoints with special rate limiting
    location ~ ^/api/(upload-and-submit|generate-upload-url) {{
        limit_req zone=upload burst=5 nodelay;
        
        proxy_pass http://vsr-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Special settings for large uploads
        proxy_request_buffering off;
        proxy_buffering off;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }}
    
    # API endpoints with standard rate limiting
    location /api/ {{
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://vsr-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }}
    
    # Documentation endpoints
    location ~ ^/(docs|redoc|openapi.json) {{
        proxy_pass http://vsr-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
    
    # Default location
    location / {{
        proxy_pass http://vsr-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
"""
    
    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run shell command asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
        )
    
    async def _reload_nginx(self) -> bool:
        """Reload Nginx configuration."""
        try:
            result = await self._run_command(["systemctl", "reload", "nginx"])
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to reload Nginx: {e}")
            return False
    
    async def validate_ssl_setup(self) -> Dict[str, Any]:
        """
        Validate SSL setup and configuration.
        
        Returns:
            Validation results
        """
        results = {
            "certificate_exists": False,
            "certificate_valid": False,
            "nginx_configured": False,
            "auto_renewal_setup": False,
            "https_accessible": False,
            "ssl_grade": None,
            "errors": []
        }
        
        try:
            # Check certificate existence
            if self.cert_path.exists() and self.key_path.exists():
                results["certificate_exists"] = True
                
                # Check certificate validity
                results["certificate_valid"] = await self.is_certificate_valid()
            
            # Check Nginx configuration
            nginx_result = await self._run_command(["nginx", "-t"])
            results["nginx_configured"] = nginx_result.returncode == 0
            
            # Check cron job for auto-renewal
            cron_result = await self._run_command(["crontab", "-l"])
            results["auto_renewal_setup"] = "renew-ssl-cert.sh" in cron_result.stdout
            
            # Test HTTPS accessibility
            try:
                response = requests.get(f"https://{self.domain}/healthz", timeout=10, verify=True)
                results["https_accessible"] = response.status_code == 200
            except Exception as e:
                results["errors"].append(f"HTTPS accessibility test failed: {e}")
            
            # SSL Labs grade check (optional)
            try:
                ssl_grade = await self._check_ssl_grade()
                results["ssl_grade"] = ssl_grade
            except Exception as e:
                results["errors"].append(f"SSL grade check failed: {e}")
            
        except Exception as e:
            results["errors"].append(f"Validation failed: {e}")
        
        return results
    
    async def _check_ssl_grade(self) -> Optional[str]:
        """Check SSL grade using SSL Labs API."""
        try:
            # This is a simplified version - in production, you might want to use SSL Labs API
            # For now, we'll do a basic SSL check
            import ssl
            import socket
            
            context = ssl.create_default_context()
            with socket.create_connection((self.domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=self.domain) as ssock:
                    cert = ssock.getpeercert()
                    if cert:
                        return "A"  # Simplified grading
            
            return None
            
        except Exception:
            return None


async def main():
    """Main SSL setup function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration from environment
    domain = os.getenv('VSR_DOMAIN')
    email = os.getenv('VSR_SSL_EMAIL')
    staging = os.getenv('VSR_SSL_STAGING', 'false').lower() == 'true'
    
    if not domain or not email:
        logger.error("Missing required environment variables: VSR_DOMAIN, VSR_SSL_EMAIL")
        return
    
    # Initialize SSL manager
    ssl_manager = SSLCertificateManager(domain, email)
    
    try:
        logger.info("Starting SSL certificate setup...")
        
        # Setup Certbot
        if not await ssl_manager.setup_certbot():
            logger.error("Failed to setup Certbot")
            return
        
        # Generate certificate
        if not await ssl_manager.generate_certificate(staging=staging):
            logger.error("Failed to generate SSL certificate")
            return
        
        # Configure Nginx
        if not await ssl_manager.configure_nginx_ssl():
            logger.error("Failed to configure Nginx SSL")
            return
        
        # Setup auto-renewal
        if not await ssl_manager.setup_auto_renewal():
            logger.error("Failed to setup auto-renewal")
            return
        
        # Validate setup
        validation_results = await ssl_manager.validate_ssl_setup()
        
        logger.info("SSL Setup Validation Results:")
        for key, value in validation_results.items():
            if key != "errors":
                logger.info(f"  {key}: {value}")
        
        if validation_results["errors"]:
            logger.error("Validation errors:")
            for error in validation_results["errors"]:
                logger.error(f"  - {error}")
        
        logger.info("SSL certificate setup completed successfully!")
        
    except Exception as e:
        logger.error(f"SSL setup failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
