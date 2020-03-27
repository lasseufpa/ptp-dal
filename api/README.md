# PTP API

This API is used to facilitate the download and search of the datasets acquired
with our FPGA-based testbed.

## Credentials

### Create the client key and CSR
This API uses mutual SSL authentication, on which both client and server authenticate each other through digital certificates. By doing so, both parties can verify each other's identity. To use this service, you need to obtain a valid client certificate, signed by LASSE's certification authority (CA). The procedure is as follows:

1. You generate a private/public key pair.
2. You generate a certificate signing request (CSR), which contains your public key and is signed using your private key.
3. You send the CSR to us (LASSE).
4. We sign your CSR using our CA's private key.
5. We send back the final digital certificate that you are going to use as a client to the dataset server. This is the CA-signed digital certificate of your identity.

To generate your key and CSR, run the following commands:

> Note: Currently the package `requests` [does not support encrypted
> keyfiles](https://requests.readthedocs.io/en/master/user/advanced/#client-side-certificates).
> Hence, we shall generate a non-encrypted key.

```bash
# Client key
openssl genrsa -out <your_name>.key 4096

# CSR to obtain certificate
openssl req -new -key <your_name>.key -out <your_name>.csr
```

After that, send the `<your_name>.csr` to the project coordinator in order to
receive your valid certificate.


## Endpoints

#### Datasets download
GET: `https://ptp.database.lasseufpa.org/api/dataset/<dataset_name>`

#### Datasets search
POST: `https://ptp.database.lasseufpa.org/api/search`

