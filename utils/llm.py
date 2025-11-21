def chat(content):
    import httpx

    response = httpx.post(
        url='https://openrouter.ai/api/v1/chat/completions',
        headers={
            'Authorization': 'Bearer sk',
            'Content-Type': 'application/json'
        },
        json={
            'model': 'minimax/minimax-m2:free',
            'messages': [
                {
                    'role': 'user',
                    'content': content
                }
            ],
            'reasoning':{'enable':False}
        }
    )

    return response.json()['choices'][0]['message']
    # return completion.choices[0].message.content


content = input('Enter your message: ')

print(chat(content))

