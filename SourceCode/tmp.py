def tra_ve_value(name, team):
    import requests

    def xy_ly_ten_1(name, team):
        res1 = list(name.split()) + list(team.split())
        return "%20".join(res1)

    def xy_ly_ten_2(name, team):
        
        res1 = list(name.lower().split()) + list(team.lower().split())
        return " ".join(res1)

    link = 'https://www.footballtransfers.com/us/search/actions/search'
    name_header = xy_ly_ten_1(name, team)
    name_payload = xy_ly_ten_2(name, team)

    headers = {
        'authority': 'www.footballtransfers.com',
        'accept': '*/*',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'referer': 'https://www.footballtransfers.com/us/search?search=' + name_header,
        'user-agent': 'Mozilla/5.0'
    }

    params = {
        'search_page': 1,
        'search_value': name_payload,
        'players': 1,
        'teams': 1,
    }

    response = requests.post(link, headers=headers, data=params)
    data=response.json()
    if data.get('found',0) and data.get("hits"):
        value=data['hits'][0]['document'].get("transfer_value")
        return value
    return None
