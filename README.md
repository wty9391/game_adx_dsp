# game_adx_dsp


# 字段含义

## adx

### Scala指标
* T_abort: 本轮流拍数
* T_abort_click: 本轮流拍点击数
* T_abort_proportion: 本轮流拍比率
* T_bid_request: 本轮总竞价请求数
* T_impression: 本轮展现数=T_bid_request-T_abort
* T_revenue: 本轮收益
* T_revenue_gain: 本轮通过设置保留价而增加的收益
* reserve_price: 本轮保留价


* abort: 总流拍数
* abort_click: 总流拍点击数
* abort_proportion: 总流拍比率
* bid_request: 总竞价请求数
* impression: 总展现数
* revenue: 总收益
* revenue_gain: 总通过设置保留价而增加的收益

### image指标
* highest_bids: 最高出价
* market_prices: 市场价
* second_highest_bids: 第二高出价
* pctrs: 预估点击率

## dsp
### Scala指标

* T_click: 本轮点击数
* T_cost: 本轮广告消耗
* T_impression: 本轮展现数

* available_budget: 可用预算
* budget: 总预算
* click: 总点击数
* cost: 总广告消耗
* impression: 总广告展现数
* bidding_para: 竞价参数，b= α * θ 中的α，其中θ指预估CTR


### image指标
* pctrs: 预估点击率
* bids: 出价
* market_prices: 所观测到的成交价