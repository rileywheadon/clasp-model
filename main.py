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

    # Fit the intracellular model and plot the results
    bl_models = im.intracellular_model()
    im.plot_bl_functions(bl_models)
    im.plot_intracellular_signalling(bl_models, "Hill (2)")

    # Run the unmodified cell column model
    print("\nRunning unmodified cell column model. This may take a while.")
    cm.fit_model()

    # Run the modified cell column model
    print("\nRunning modified cell column model. This may take a while.")
    cm.fit_model(modified=True)

    # Run the modified cell column model on atrichoblast data
    print("\nRunning cell column model on atrichoblast data. This may take a while.")
    cm.fit_model(cell_type="atrichoblast", modified=True)

main()
