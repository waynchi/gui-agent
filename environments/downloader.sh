#!/bin/bash

# URL of the Google Drive file
FILE_ID="1gxXalk9O0p9eu1YkIJcmZta1nvvyAJpA"
OUTPUT="downloaded_file.tar.gz"  # Change this to the desired output file name

# Download the file
gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$OUTPUT"

# Untar the file
tar -xzvf "$OUTPUT"

docker load --input shopping_admin_final_0719.tar
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
# wait ~1 min to wait all services to start

docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://<your-server-hostname>:7780" # no trailing slash
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://<your-server-hostname>:7780/" WHERE path = "web/secure/base_url";'
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush