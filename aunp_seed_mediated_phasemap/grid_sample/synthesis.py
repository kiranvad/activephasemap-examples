# %%
from opentrons.simulate import get_protocol_api
import json 
import pandas as pd 

protocol = get_protocol_api('2.18')
CUSTOM_LABWARE_PATH = "../opentrons_custom_labware/"
protocol.home()

# %%
# load all the labware modules
tiprack_300 = protocol.load_labware(
    load_name="opentrons_96_tiprack_300ul",
    location=1)

p300 = protocol.load_instrument(
    instrument_name="p300_single_gen2",
    mount="right",
    tip_racks=[tiprack_300]
    )

tiprack_20 = protocol.load_labware(
    load_name="opentrons_96_tiprack_20ul",
    location=2)

p20 = protocol.load_instrument(
    instrument_name="p20_single_gen2",
    mount="left",
    tip_racks=[tiprack_20]
    )

with open(CUSTOM_LABWARE_PATH+'20mlscintillation_12_wellplate_18000ul.json') as labware_file:
    stocks_def = json.load(labware_file)
    stocks = protocol.load_labware_from_definition(stocks_def, location=3)

plate = protocol.load_labware(
    load_name="corning_96_wellplate_360ul_flat",
    location=4)

# %%
tiprack_20_wells = [well for row in tiprack_20.rows() for well in row]
tiprack_300_wells = [well for row in tiprack_300.rows() for well in row]
stocks_wells = [well for row in stocks.rows() for well in row]
plate_wells = [well for row in plate.rows() for well in row]

# %%
def synthesize(stock_index, ds):
    """ Synthesize AuNP by mixing components

    stock_index : index of the stock to add (int)
    ds : a pandas dataseries with volumes to be added. 
    """
    p20.pick_up_tip(tiprack_20_wells[int(stock_index)])
    p300.pick_up_tip(tiprack_300_wells[int(stock_index)])
    has_used_p20, has_used_p300 = False, False
    for index, value in ds.items():
        if value<20:
            pipette = p20 
            has_used_p20 = True
        else:
            pipette = p300
            has_used_p300 = True
        source_well = stocks_wells[int(stock_index)]
        target_well = plate_wells[int(index)]
        print("Dispensing %s of %d from %s into well %s "%(ds.name, value, source_well.well_name, target_well.well_name)) #,end='\r', flush=False)
        pipette.aspirate(value, source_well)
        pipette.dispense(value, target_well)
        if not stock_index==0:
            pipette.blow_out()

    if has_used_p20:
        p20.drop_tip()
    else:
        p20.return_tip()

    if has_used_p300:
        p300.drop_tip()
    else:
        p300.return_tip()

# %%
volume_df = pd.read_csv('./grid_volumes.csv')
for stock_index,(_, stock_vol_series)  in enumerate(volume_df.items()):
    synthesize(stock_index, stock_vol_series)

# %%


# %%



