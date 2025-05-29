# %%
from opentrons.simulate import get_protocol_api

import json 
import pandas as pd 
from datetime import datetime
from pytz import timezone 
Pacific = timezone('US/Pacific')
from IPython.display import clear_output
import time

# %%
START_WELL = "A1"
volume_df = pd.read_csv('./misc/repeat_96_well_plate.csv')
volume_df

# %%
protocol = get_protocol_api('2.18')
CUSTOM_LABWARE_PATH = "../opentrons_custom_labware/"
protocol.home()

# %%
protocol.set_rail_lights(True)

# %%
# load all the labware modules
tiprack_300 = protocol.load_labware(
    load_name="opentrons_96_tiprack_300ul",
    location=7)

p300 = protocol.load_instrument(
    instrument_name="p300_single_gen2",
    mount="right",
    tip_racks=[tiprack_300]
    )

tiprack_20 = protocol.load_labware(
    load_name="opentrons_96_tiprack_20ul",
    location=4)

p20 = protocol.load_instrument(
    instrument_name="p20_single_gen2",
    mount="left",
    tip_racks=[tiprack_20]
    )

with open(CUSTOM_LABWARE_PATH+'20mlscintillation_12_wellplate_18000ul.json') as labware_file:
    stocks_def = json.load(labware_file)
    stocks = protocol.load_labware_from_definition(stocks_def, location=1)

temp_module = protocol.load_module(module_name="temperature module gen2", location=6)
alplate_adapter = temp_module.load_adapter("opentrons_aluminum_flat_bottom_plate")
plate = alplate_adapter.load_labware('corning_96_wellplate_360ul_flat')
temp_module.set_temperature(celsius=30)

# %%
tiprack_300.set_offset(x=0.50, y=1.10, z=0.00)
tiprack_20.set_offset(x=0.00, y=0.90, z=0.00)
plate.set_offset(x=0.00, y=1.50, z=0.00)

# %%
tiprack_20_wells = [well for row in tiprack_20.rows() for well in row]
tiprack_300_wells = [well for row in tiprack_300.rows() for well in row]
stocks_wells = [well for row in stocks.rows() for well in row]
plate_wells = [well for row in plate.rows() for well in row]

# %%
START_WELL_INDEX = next((i for i, well in enumerate(plate_wells) if well.well_name == START_WELL), None)
if START_WELL_INDEX is not None:
    print(f"The index of well {START_WELL} is {START_WELL_INDEX}.")
else:
    print(f"Well {START_WELL} not found.")

# %%
def synthesize(stock_index, ds):
    """ Synthesize AuNP by mixing components

    stock_index : index of the stock to add (int)
    ds : a pandas dataseries with volumes to be added. 
    """
    p300.pick_up_tip(tiprack_300_wells[int(stock_index)])
    has_used_p20, has_used_p300 = False, False
    need_p20 = (ds<20).any()
    if need_p20:
        p20.pick_up_tip(tiprack_20_wells[int(stock_index)])
    for index, value in ds.items():
        if value>20:
            pipette = p300 
            has_used_p300 = True
        else:
            pipette = p20
            has_used_p20 = True
        source_well = stocks_wells[int(stock_index)]
        target_well = plate_wells[int(index) + START_WELL_INDEX]
        current_date_time = datetime.now(Pacific)
        time_str = current_date_time.strftime('%H:%M:%S')
        print("[%s] Dispensing %s of %d from %s into well %s "%(time_str, ds.name, value, source_well.well_name, target_well.well_name)) #,end='\r', flush=False)
        pipette.aspirate(value, source_well)
        pipette.dispense(value, target_well.top())
        if not stock_index in [0, 5]:
            pipette.blow_out()
        if stock_index==1:
            has_used_p300 = False
            
    if has_used_p20:
        p20.drop_tip()
        
    if has_used_p300:
        p300.drop_tip()
    else:
        p300.return_tip()

# %%
for _ in range(2):
    synth_volumes_df = volume_df.sample(frac=1)
    for stock_index,(_, stock_vol_series) in enumerate(synth_volumes_df.items()):
        synthesize(stock_index, stock_vol_series)

# Add water for control
p300.pick_up_tip(tiprack_300_wells[1])
source_well = stocks_wells[1]
p300.aspirate(300, stocks_wells[1])
target_well = plate_wells[len(volume_df) + START_WELL_INDEX]
p300.dispense(200, target_well.top())
current_date_time = datetime.now(Pacific)
time_str = current_date_time.strftime('%H:%M:%S')
print("[%s] Dispensing %s of %d from %s into well %s "%(time_str, "water", 300, source_well.well_name, target_well.well_name))
p300.drop_tip()


# %%
current_date_time = datetime.now(Pacific)
print(current_date_time.strftime('End Time: %H:%M:%S'))

# %%
protocol.set_rail_lights(False)

# %%
protocol.home()

# %%
print("Keeping the temperature module on for 30 mins at the initial temperature")

# %%
def countdown(minutes):
    total_seconds = minutes * 60
    for remaining in range(total_seconds, 0, -1):
        # Convert seconds to minutes and seconds
        mins, secs = divmod(remaining, 60)
        # Clear output in the notebook for a clean update
        clear_output(wait=True)
        print(f"Time remaining: {mins:02d}:{secs:02d}")
        time.sleep(1)
    clear_output(wait=True)
    print("Time's up!")

# Start a 30-minute countdown
countdown(30)


# %%



