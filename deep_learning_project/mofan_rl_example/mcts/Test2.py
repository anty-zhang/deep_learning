

s = """insert overwrite table map_test.zgq_order_detail_0716 partition (h='{}')
select a.order_id, a.trace_id, a.od, b.od, b.match_link,a.req_time, a.subgraph from
(select param['order_id'] as order_id, parse(param['subgraph']) as subgraph, param['trace_id'] as trace_id, param['req_time'] as req_time,
param['od'] as od from didimap.ods_log__map_route_ranker_detail where year = '2018' and month='07' and day = '16' and hour='{}' ) a
left join
(select param['order_id'] as order_id, param['traceid'] as trace_id, param['od'] as od,param['match_link'] as match_link from didimap.ods_log__map_route_ranker where year = '2018' and month='07' and day = '16' and hour='{}' and param['match_link']=2) b
on (a.order_id=b.order_id and a.trace_id=b.trace_id)
where b.match_link <> '' and a.order_id > '0';


insert overwrite table map_test.zgq_subgraph_0716 partition (h='{}')
select order_id, subgraph from (select order_id, subgraph,row_number() over (partition by order_id order by req_time) index
from map_test.zgq_order_detail_0716 where h='{}') t where  t.index=1;"""

s1 = """insert overwrite table map_test.zgq_routego_new_0716 partition(h='{i}')
select a.order_id,a.restore,a.start_pos,a.end_pos,a.real_route, b.subgraph from
(select * from map_test.zgq_routego_old_0716 where h ='{i}') a
left join
(select * from map_test.zgq_subgraph_0716 where h = '{i}' ) b
on (a.order_id = b.order_id);"""

s2 = """insert overwrite table map_test.zgq_routego_old_0716_1_res partition (h='{i}')
select a.order_id,a.restore,a.start_pos,a.end_pos,a.real_route,b.subgraph from
(select * from map_test.zgq_routego_old_0716_1 where h = '{i}' ) a
left join
(select * from map_test.zgq_subgraph_0716 where h = '{i}') b
on (a.order_id = b.order_id);"""
l = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', ]
day = ['10', '11', '12', '13', '14', '15', '16']

s4 = """insert overwrite table map_test.zgq_subgraph_0710_0716 partition(day='{day}{hour}')
select param['order_id'], collect_set(parse(param['subgraph']))[0] from didimap.ods_log__map_route_ranker_detail
where year = '2018'  and month = '07' and day = '{day}' and hour = '{hour}' and param['order_id'] > '0'
group by param['order_id'];"""


# for d in day:
#     for h in l:
#         print(s4.format(hour=h, day=d))


s = "17617413623935|0.10307169139197499|TWpnNE5UZzFORFkyTXpJMU5qVXhOVGd6_1543581082205||114.10688,22.549706;114.107735,22.549948;114.107735,22.549948;114.10839,22.550161;114.10839,22.550161;114.10927,22.550413;114.10936,22.550438;114.10936,22.550438;114.110115,22.550686;114.110115,22.550686;114.110176,22.550707|1543581082,114.10688,22.549706,30.0,0.0,-1.0,0;1543581085,114.10688,22.549706,30.0,0.0,-1.0,0;1543581088,114.10688,22.549706,30.0,0.0,-1.0,0;1543581091,114.10688,22.549706,30.0,0.0,-1.0,0;1543581094,114.10688,22.549706,30.0,0.0,-1.0,0;1543581097,114.10688,22.549706,30.0,0.0,-1.0,0;1543581100,114.10688,22.549706,30.0,0.0,-1.0,0;1543581103,114.10688,22.549706,30.0,0.0,-1.0,0;1543581106,114.10688,22.549706,30.0,0.0,-1.0,0;1543581109,114.10688,22.549706,30.0,0.0,-1.0,0;1543581112,114.10688,22.549706,30.0,0.0,-1.0,0;1543581115,114.10688,22.549706,30.0,0.0,-1.0,0;1543581118,114.10688,22.549706,30.0,0.0,-1.0,0;1543581121,114.10688,22.549706,30.0,0.0,-1.0,0;1543581124,114.10688,22.549706,30.0,0.0,-1.0,0;1543581127,114.10688,22.549706,30.0,0.0,-1.0,0;1543581130,114.10688,22.549706,30.0,0.0,-1.0,0;1543581133,114.10688,22.549706,30.0,0.0,-1.0,0;1543581136,114.10688,22.549706,30.0,0.0,-1.0,0;1543581139,114.106834,22.549711,30.0,0.0,-1.0,0;1543581142,114.106834,22.549711,30.0,0.0,-1.0,0;1543581145,114.106834,22.549711,30.0,0.0,-1.0,0;1543581148,114.106804,22.54969,30.0,0.33,237.0,0;1543581151,114.10645,22.5495,30.0,3.33,244.0,0;1543581154,114.10625,22.549576,30.0,3.55,253.0,0;1543581157,114.10618,22.549557,30.0,1.22,253.0,0;1543581160,114.10588,22.549477,30.0,1.29,252.0,0;1543581163,114.105774,22.54945,30.0,0.78,256.0,0|1543581082207|1543581154222|2|10480180,10495200,956778030,956778040,621419190,621419200,10169040,90000323308421,90000323308401"

res_list = s.split("|")
traj_str = res_list[5]

print(traj_str)

traj_dict = {}
for i in traj_str.split(";"):
    arr = i.split(",")
    traj_dict[arr[0]] = arr[1] + "," + arr[2]

traj_dict_sort = [traj_dict[k] for k in sorted(traj_dict.keys())]
print(traj_dict)
print(';'.join(traj_dict_sort))



