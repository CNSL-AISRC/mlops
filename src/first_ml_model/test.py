from kserve import KServeClient
import requests
import kfp
client = kfp.Client()
print(client.get_user_namespace())

ISVC_NAME = "wine-regressor8"
kserve_client = KServeClient()

isvc_resp = kserve_client.get(ISVC_NAME)
inference_service_url = isvc_resp['status']['address']['url']
print("Inference URL:", inference_service_url)

input_data = {
    "instances": [
        [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 5, 3.51, 0.56, 9.4],
        [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 5, 3.2, 0.68, 9.8]
    ]
}

response = requests.post(f"{inference_service_url}/v1/models/{ISVC_NAME}:predict", json=input_data)
print(response.text)