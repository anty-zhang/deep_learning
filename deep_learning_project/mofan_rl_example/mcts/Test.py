

# s = "route= 1,2,3,4,5,6,7"
route_input = "6863930,90000260498780,90000260412950,5765410,5765420,877579610,877579620,491431860,6647991,422366081,422366071,6714661,422366201,422366191,90000285124710,90000285201350,132084800,622185390,422361390,422361400,6808220,6809490,6809500,5849390,5791461,6714681,877579451,877579441,6716081,5762391,422362640,422362650,422362630,6738250,132085000,132085010,6787611,422362531,422362521,7008661,132085031,132085021,7008701,7008691,6643861,6799400,6799410,126347950,126347960,90000242392280,90000242392200,90000242392320,90000242315780,300503480,300503490,152550970,152550990,152551010,152551030,152551050,6629510,90000284502220,90000284359420,6823670,6823690,6823660,6823610,300503770,300503790,300503800,6629470,6629410,6629420,6823560,6824140,6824160,701371610,701371620,6808920,6808940,6808970,6659330,622586230,622586240,5823321,6839311,6671701,6671691,5747601747601,5747611,6846120,6846130,5791380,5777010,6678250,6644370,6644380,6856660,6677690,6644420,6644430,6677730,6677740,6677790,500623910,500623920,6677750,501738330,501738340,6858750,6858760"

# comm = "1000,2,200,5,10001"
# comm = "1000,2,3,4,5,10001"
link_input = "6669771,6669791,6669781,5777051,5823321,6839311,6671701,6671691,5747601"


def replace_route(link, route):
    comm_list = link.split(",")
    route_list = route.split(",")
    flag = False
    for i in range(len(comm_list)):
        replace_start_index = -1
        replace_end_index = -1
        if comm_list[i] in route_list:
            replace_start_index = route_list.index(comm_list[i])

        check_end_index = -1
        if i != len(comm_list):
            for i_r in reversed(range(len(comm_list))):
                if i_r > i and comm_list[i_r] in route_list:
                    replace_end_index = route_list.index(comm_list[i_r])
                    check_end_index = i_r
                    break

        if (replace_end_index > replace_start_index) and (replace_start_index > 0) and (replace_end_index > 0):
            replace_str = ','.join(comm_list[i:check_end_index+1])
            be_replace_str = ','.join(route_list[replace_start_index:replace_end_index+1])
            if be_replace_str.strip() != replace_str.strip():
                route = route.replace(be_replace_str, replace_str)
                flag = True
            # print("replace_str: ", replace_str)
            # print("be_replace_str: ", be_replace_str)
            # print("s: ", s)
            break

    return route, flag


r, f = replace_route(link_input, route_input)
print("r=", r)
print("f=", f)

