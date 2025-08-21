# DigitalOcean Spaces Lifecycle Policies

This document outlines the lifecycle policies for the DigitalOcean Spaces bucket used in the Video Subtitle Removal API.

## Bucket Structure

The VSR API uses the following bucket structure:

- **Bucket Name**: `vsr-videos`
- **Folders**:
  - `uploads/YYYYMMDD/<uuid>/video.mp4` - Raw uploaded videos
  - `processed/YYYYMMDD/<uuid>/video.mp4` - Processed videos with subtitles removed
  - `logs/` - Processing logs and other system logs

## Lifecycle Policies

The following lifecycle policies should be configured in the DigitalOcean Spaces control panel:

### 1. Uploads Folder (7-day retention)

- **Prefix**: `uploads/`
- **Action**: Expire (Delete)
- **Days After Creation**: 7

### 2. Processed Folder (30-day retention)

- **Prefix**: `processed/`
- **Action**: Expire (Delete)
- **Days After Creation**: 30

### 3. Logs Folder (90-day retention)

- **Prefix**: `logs/`
- **Action**: Expire (Delete)
- **Days After Creation**: 90

## Configuration Steps

To configure these lifecycle policies in the DigitalOcean Spaces control panel:

1. Log in to the DigitalOcean Control Panel
2. Navigate to Spaces
3. Select the `vsr-videos` bucket
4. Click on the "Settings" tab
5. Scroll down to "Lifecycle Rules"
6. Click "Add a Rule"
7. Configure each rule as specified above
8. Save the rules

## Verification Checklist

To verify that the lifecycle policies are working correctly:

1. Create test objects in each folder with known creation dates
2. Monitor the objects as they approach their expiration dates
3. Verify that objects are deleted after the specified retention period
4. Document any issues or discrepancies

## Important Notes

- The application code should not delete objects prematurely before their lifecycle expiration
- Always use the appropriate folder structure when generating keys for uploads and processed videos
- The lifecycle policies only apply to objects in the specified folders
- Objects outside of these folders will not be automatically deleted

## MinIO Local Development

For local development with MinIO, you can configure similar lifecycle policies using the MinIO Client (mc) command-line tool:

```bash
# Configure MinIO client
mc alias set minio http://localhost:9000 minioadmin minioadmin

# Add lifecycle rules
mc ilm add --expiry-days 7 --prefix "uploads/" minio/vsr-videos
mc ilm add --expiry-days 30 --prefix "processed/" minio/vsr-videos
mc ilm add --expiry-days 90 --prefix "logs/" minio/vsr-videos

# List lifecycle rules
mc ilm ls minio/vsr-videos
```

## Testing Lifecycle Rules

You can test the lifecycle rules using the MinIO Client:

```bash
# Test lifecycle rules (dry run)
mc ilm rule --expiry-days 7 --prefix "uploads/" ls minio/vsr-videos
```

This will show which objects would be affected by the rule without actually applying it.
