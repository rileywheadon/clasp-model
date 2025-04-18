# This file runs all simulations and generates all figures used in the manuscript
import exploratory_analysis as ea
import intracellular_model as im
import column_model as cm

def main():

    # Exploratory data analysis on trichoblast data
    ea.plot_unprocessed_data()
    ea.plot_processed_data()
    ea.plot_binned_data()

    # Exploratory data analysis on atrichoblast data
    ea.plot_unprocessed_data(cell_type = "atrichoblast")
    ea.plot_processed_data(cell_type = "atrichoblast")
    ea.plot_binned_data(cell_type = "atrichoblast")

    # Fit the intracellular model
    model_linear, model_hill, model_quadratic = im.simulate_bl_functions()
    intracellular_params = [round(float(x), 3) for x in model_linear[2]]

    # Simulate the best intracellular model on the mutants
    im.simulate_intracellular_signalling(model_linear, "Linear")

    # Run the unmodified cell column model
    print("\nRunning unmodified cell column model. This may take a while.")
    cm.fit_model(intracellular_params)

    # Run the modified cell column model
    print("\nRunning modified cell column model. This may take a while.")
    cm.fit_model(intracellular_params, modified=True)

    # Run the modified cell column model on atrichoblast data
    print("\nRunning cell column model on atrichoblast data. This may take a while.")
    cm.fit_model(intracellular_params, cell_type="atrichoblast", modified=True)

main()
