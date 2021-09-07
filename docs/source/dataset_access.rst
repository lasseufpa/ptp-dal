Dataset Access
=======================================

The datasets acquired with the FPGA-based testbed are kept within the
PTP dataset database (DB). These are accessible through our PTP dataset
API hosted at https://ptp.database.lasseufpa.org/api/.

This API uses mutual SSL authentication, on which both client and server
authenticate each other through digital certificates. Hence, to use this
service, you need to obtain a valid client certificate signed by our
certification authority (CA).

If you are interested in accessing our datasets, please follow the
procedure below:

1. Generate a private/public key pair.

.. code:: bash

    # Client key
    openssl genrsa -out <your_name>.key 4096

2. Generate a certificate signing request (CSR), which contains your
   public key and is signed using your private key.

.. code:: bash

    # CSR to obtain certificate
    openssl req -new -key <your_name>.key -out <your_name>.csr

3. Send the CSR to us at ptp.dal@gmail.com and let us know the network
   scenarios or types of datasets you are interested in exploring.

4. We sign your CSR and send you the final (CA-signed) digital
   certificate that you will use to access the dataset DB API.

5. Try accessing the dataset API:

First, run:

::

    ./dataset.py search

The application will prompt you for access information. When asked about
``Download via API or SSH?``, reply with ``API`` (or just press enter to
accept the default response). Next, fill in the paths to your private
key (generated in step 1) and the digital certificate (obtained in step
4).

After that, the command should return the list of datasets.

API Endpoints
~~~~~~~~~~~~~

**Dataset download:** GET:
``https://ptp.database.lasseufpa.org/api/dataset/<dataset_name>``

**Dataset search** POST:
``https://ptp.database.lasseufpa.org/api/search``