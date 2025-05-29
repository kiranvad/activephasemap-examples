metadata = {'apiLevel':'2.19'}

def run(protocol):
    # Load temperature module and add AL plate adapter
    temp_module = protocol.load_module(module_name="temperature module gen2", location=6)
    alplate_adapter = temp_module.load_adapter("opentrons_aluminum_flat_bottom_plate")
    wellplate = alplate_adapter.load_labware('corning_96_wellplate_360ul_flat')

    # Load pipette tips
    tiprack_300 = protocol.load_labware('opentrons_96_tiprack_300ul', 7)
    tiprack_20 = protocol.load_labware('opentrons_96_tiprack_20ul', 4)

    # Load stock solutions
    stocks = protocol.load_labware('20mlscintillation_12_wellplate_18000ul', 1)

    # Load pipettes
    P300 = protocol.load_instrument(instrument_name = 'p300_single_gen2', mount = 'right', tip_racks=[tiprack_300])
    P20 = protocol.load_instrument(instrument_name = 'p20_single_gen2', mount = 'left', tip_racks=[tiprack_20])

    P300.pick_up_tip()
    P300.aspirate(100, stocks["A1"])
    P300.dispense(100, wellplate["A1"])
    P300.return_tip()

    P20.pick_up_tip()
    P20.aspirate(10, stocks["A1"])
    P20.dispense(10, wellplate["A1"])
    P20.return_tip()
