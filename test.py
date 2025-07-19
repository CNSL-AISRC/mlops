import kfp
client = kfp.Client(host='http://10.5.110.131:31047', namespace='admin', credentials='~/.kube/config')

print(client.list_experiments())

print(client.list_runs())