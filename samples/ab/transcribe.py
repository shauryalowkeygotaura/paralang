import os, sys, json, glob, uuid, urllib.request
for f in sorted(glob.glob(sys.argv[1])):
    b = "----" + uuid.uuid4().hex
    head = (f"--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode()
    body = head + open(f, "rb").read() + f"\r\n--{b}--\r\n".encode()
    r = urllib.request.Request("https://api.groq.com/openai/v1/audio/transcriptions", data=body,
        headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"],
                 "Content-Type": "multipart/form-data; boundary=" + b, "User-Agent": "curl/8"})
    print(f"{f:30} | {json.load(urllib.request.urlopen(r, timeout=60))['text'].strip()}")
