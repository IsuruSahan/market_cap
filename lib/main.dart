import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart ' as http;

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  State<StatefulWidget> createState() => MyHomePage();
}

class MyHomePage extends State<MyApp> {
  var list;
  var refreshKey = GlobalKey<RefreshIndicatorState>();

  @override
  void initState() {
    super.initState();
    refreshListCoin();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'COIN TRACKER',
      theme: ThemeData.light(),
      home: Scaffold(
        appBar: AppBar(title: Text('COIN TRACKER')),
        body: Center(
          child: RefreshIndicator(
            key: refreshKey,
            child: FutureBuilder<List<CoinMarket>>(
              future: list,
              builder: (context, snapshot) {
                if (snapshot.hasData) {
                  List<CoinMarket> coins = snapshot.data;

                  return ListView(
                    children: coins
                        .map((coin) => Column(
                              mainAxisAlignment: MainAxisAlignment.start,
                              children: <Widget>[
                                Row(
                                  children: <Widget>[
                                    Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: <Widget>[
                                        Container(
                                          padding: const EdgeInsets.only(
                                              left: 8.0, bottom: 8.0),
                                          child: Image.network(
                                              'https://res.cloudinary.com/dxi90ksom/image/upload/${coin.symbol.toLowerCase()}.png'),
                                          width: 56.0,
                                          height: 56.0,
                                        )
                                      ],
                                    ),
                                    Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: <Widget>[
                                          Container(
                                            padding: const EdgeInsets.all(4.0),
                                            child: Text(
                                                '${coin.symbol} | ${coin.name}'),
                                          )
                                        ]),
                                    Expanded(
                                      child: Container(
                                        child: Column(
                                          crossAxisAlignment:
                                              CrossAxisAlignment.end,
                                          children: <Widget>[
                                            Container(
                                              padding:
                                                  const EdgeInsets.all(8.0),
                                              child: Text(
                                                  '\$${double.parse(coin.price).toStringAsFixed(2)}'),
                                            )
                                          ],
                                        ),
                                      ),
                                    )
                                  ],
                                ),
                                Container(
                                  padding: const EdgeInsets.all(8.0),
                                  child: Row(
                                    mainAxisAlignment:
                                        MainAxisAlignment.spaceBetween,
                                    children: <Widget>[
                                      Text('${coin.currency}')
                                    ],
                                  ),
                                )
                              ],
                            ))
                        .toList(),
                  );
                } else if (snapshot.hasError) {
                  Text('Error while loadig coin list: ${snapshot.error}');
                }

                return new CircularProgressIndicator();
              },
            ),
            onRefresh: refreshListCoin,
          ),
        ),
      ),
    );
  }

  Future<Null> refreshListCoin() {
    refreshKey.currentState?.show(atTop: false);

    setState(() {
      list = fetchListCoin();
    });

    return null;
  }
}

Future<List<CoinMarket>> fetchListCoin() async {
  final api_endpoint = await http
      .get(Uri.https('https://api.nomics.com/v1/currencies/ticker/', 'list/1'));

  if (api_endpoint.statusCode == 200) {
    List coins = json.decode(api_endpoint.body);
    return coins.map((coin) => new CoinMarket.fromjson(coin)).toList();
  } else
    throw Exception('failed to load Coin list');
}

class CoinMarket {
  final String id;
  final String currency;
  final String price;
  final String name;
  final String symbol;

  CoinMarket({this.id, this.currency, this.price, this.name, this.symbol});

  factory CoinMarket.fromjson(Map<String, dynamic> json) {
    return CoinMarket(
      id: json['id'],
      currency: json['currency'],
      price: json['price'],
      name: json['name'],
      symbol: json['symbol'],
    );
  }
}
