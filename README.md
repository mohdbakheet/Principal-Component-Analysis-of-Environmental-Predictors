# Principal-Component-Analysis-of-Environmental-Predictors

Below is a step‐by‐step workflow (in R) for using PCA to identify and reduce multicollinearity among our environmental predictors, then selecting a subset of variables (or principal components) for downstream modeling. 

---

## 1. Prepare and extract predictor data

1. **Load your raster layers** (water, climate, land‐use, population density, etc.) into a single `RasterStack` (or `SpatRaster` if you're using terra). For example:

   ```r
   library(terra)   # or library(raster)

   # Example using terra:
   water_ras    <- rast("path/to/water_layer.tif")
   climate_ras1 <- rast("path/to/temperature.tif")
   climate_ras2 <- rast("path/to/precipitation.tif")
   landuse_ras  <- rast("path/to/landuse.tif")
   popdens_ras  <- rast("path/to/population_density.tif")

   # Stack them together (ensure they have same CRS & extent/resolution)
   preds_stack <- c(water_ras, climate_ras1, climate_ras2, landuse_ras, popdens_ras)
   names(preds_stack) <- c("water", "temp", "precip", "landuse", "popdens")
   ```

2. **Mask or crop** the stack to your study area (optional, but reduces computation time):

   ```r
   # Suppose you have a shapefile or extent for your study area:
   study_area <- vect("path/to/study_area.shp")
   preds_stack <- crop(preds_stack, study_area)
   preds_stack <- mask(preds_stack, study_area)
   ```

3. **Sample raster values** to a data.frame. PCA requires a matrix/data.frame where rows are observations and columns are variables. Typically, you either

   * sample a random subset of pixels (e.g. 10,000–50,000) to represent global variation, or
   * sample at your presence/background or absence points if doing species‐distribution–type modeling.

   ```r
   set.seed(42)
   # Sample 20,000 random cells (na.rm=TRUE removes cells where any layer is NA)
   samp_df <- as.data.frame(terra::spatSample(preds_stack,
                                              size = 20000,
                                              as.data.frame = TRUE,
                                              xy = FALSE,
                                              na.rm = TRUE))
   # Check:
   head(samp_df)
   #    water    temp  precip landuse popdens
   # 1   0.12 15.3456 1200.2       2   45.33
   # 2   0.56 13.6789  950.6       1   67.12
   # ...
   ```

4. **(Optional) Remove categorical variables** or encode them appropriately. If `landuse` is a categorical raster (e.g. integers representing classes), you have two options:

   * Exclude it from this PCA (perform PCA only on continuous predictors), or
   * One‐hot‐encode land‐use categories before PCA (but that can blow up dimensionality).

   For simplicity, let's assume “landuse” is numeric (e.g. percent water–agriculture), or else we drop it:

   ```r
   samp_df_cont <- samp_df[, c("water", "temp", "precip", "popdens")]
   ```

---

## 2. Standardize (center & scale) variables

PCA is sensitive to scale. Always center (mean = 0) and scale (unit variance) continuous predictors:

```r
# Compute means and sds (if you want to project new data later, save these)
library(stats)
means <- apply(samp_df_cont, 2, mean, na.rm = TRUE)
sds   <- apply(samp_df_cont, 2, sd,   na.rm = TRUE)

samp_scaled <- scale(samp_df_cont, center = means, scale = sds)
```

---

## 3. Run PCA

Use `prcomp()` on the scaled data:

```r
pca_res <- prcomp(samp_scaled, center = FALSE, scale. = FALSE)
# (We already centered/scaled manually, so prcomp shouldn't re‐center/scale)
```

Check the basic output:

```r
summary(pca_res)
# Importance of components:
#                           PC1    PC2    PC3    PC4
# Standard deviation     1.732 1.1121 0.7543 0.3856
# Proportion of Variance 0.750 0.3090 0.1423 0.0372
# Cumulative Proportion  0.750 1.0000 1.0000 1.0000
```

* **“Standard deviation”** = square root of eigenvalues.
* **“Proportion of Variance”** tells you how much each PC explains.
* Typically, you'll look for enough PCs to explain ≥ 80–90% of variance.

---

## 4. Examine eigenvalues and decide how many PCs to keep

1. **Scree plot**:

   ```r
   # Basic scree plot:
   scree_vals <- pca_res$sdev^2
   scree_df   <- data.frame(PC = seq_along(scree_vals),
                             Variance = scree_vals / sum(scree_vals))

   barplot(scree_df$Variance,
           names.arg = paste0("PC", scree_df$PC),
           ylab = "Proportion Var.", xlab = "Principal Component",
           main = "Scree Plot")
   abline(h = 1 / ncol(samp_scaled), lty = 2, col = "grey")  # Kaiser rule threshold
   ```

2. **Cumulative variance**:

   ```r
   cumvar <- cumsum(scree_df$Variance)
   plot(seq_along(cumvar), cumvar,
        type = "b", pch = 19, xlab = "Number of PCs",
        ylab = "Cumulative Proportion Variance",
        main = "Cumulative Variance Explained")
   abline(h = 0.9, col = "red", lty = 2)
   ```

   From these plots/tables, decide to retain the first *k* PCs that explain most variance (often 2–3 PCs cover > 80%).

---

## 5. Inspect loadings to determine which variables drive each PC

The loadings (eigenvectors) tell you how each original variable contributes to each principal component.

```r
loadings <- pca_res$rotation
# Example: loadings[, 1:2] shows loadings for PC1 and PC2
round(loadings[, 1:3], 3)

#    water   temp  precip  popdens
# PC1  0.50   0.56   0.61    0.12
# PC2 -0.32   0.57   -0.10   -0.75
# PC3  0.14   -0.45   0.78    -0.41
# PC4 -0.80   -0.37   -0.11    0.43
```

* **Interpretation**:

  * Variables with large (absolute) loadings on PC1 are the ones most correlated with that axis. For instance, if PC1 loadings are high for `water`, `temp`, and `precip`, then PC1 largely represents a combined “wet‐warm” gradient.
  * PC2 might load heavily on `popdens` and `temp`, etc.

---

## 6. Decide how to select predictors

You have two main approaches:

1. **Use principal components as predictors**

   * Instead of keeping original variables, use PC1, PC2, … as orthogonal predictors in your linear model or RF.
   * Pros: eliminates multicollinearity completely.
   * Cons: hard to interpret ecologically, since each PC is a combination of variables.

2. **Select a subset of original variables based on loadings (or correlation)**

   * Identify clusters of highly correlated predictors (e.g., if `temp` and `precip` are strongly correlated, pick the one with the highest ecological relevance or the highest loading on the first PC).
   * Keep one representative from each correlated group.

Below is a simple rubric for selecting a subset:

* **Step 1:** Look at pairwise correlations (or a correlation matrix) among all continuous predictors.

  ```r
  cor_mat <- cor(samp_df_cont, use = "pairwise.complete.obs")
  print(round(cor_mat, 2))
  ```

  If any pair has |r| > 0.7–0.8, consider dropping one.

* **Step 2:** For each highly correlated pair/group, see which variable has the larger absolute loading on PC1 (or PC2) and keep that as a “proxy.”

  ```r
  # Example: if cor(temp, precip) = 0.85, check absolute loadings on PC1:
  abs(loadings[c("temp", "precip"), "PC1"])
  # Keep whichever is larger.
  ```

* **Step 3:** Repeat until you end up with an uncorrelated (or weakly correlated) set of variables.

For instance, if:

* `temp` and `precip` are correlated (r = 0.8); PC1 loading: temp = 0.56, precip = 0.61 ⇒ keep **precip**.
* `water` and `landuse` are correlated; loadings on PC1: water = 0.50, landuse=0.10 ⇒ keep **water**.
* `popdens` has low correlation with others, so keep **popdens**.

---

## 7. (Optional) Rotate PCs to aid interpretation

If your first two or three PCs are hard to interpret, you can apply a varimax rotation to the PCA loadings. This doesn't change variance explained but often yields “purer” axes:

```r
library(psych)    # for varimax
# Extract loadings for the first k PCs:
k <- 3
raw_loads <- loadings[, 1:k]  # matrix of loadings
rotated <- varimax(raw_loads)
rot_loads <- rotated$loadings  # rotated loadings

# View rotated loadings
print(round(rot_loads, 3))
```

* After rotation, you might see a “climate” axis, a “hydrology” axis, etc.
* This can help you choose which original variables best represent each rotated factor.

---

## 8. Apply selection back to full raster stack (if you want to drop layers)

Once you choose a subset of predictors (e.g., keep `precip`, `water`, `popdens`), subset your raster stack:

```r
# Suppose you decided to keep only precip, water, popdens
keep_vars <- c("precip", "water", "popdens")
reduced_stack <- preds_stack[[keep_vars]]
```

Use `reduced_stack` for any further modeling (e.g., logistic regression, random forest, MaxEnt, etc.).

---

## 9. (Alternative) Use PC rasters directly

If you prefer to model with PCs themselves rather than original variables:

1. **Project the entire raster stack onto PC space**. For each cell, compute scores of PC1, PC2, etc., using the same centering and scaling done on your sample.

   ```r
   # Function to scale a raster brick (using saved means & sds):
   scale_raster <- function(rast_layer, mean_val, sd_val) {
     (rast_layer - mean_val) / sd_val
   }

   # Scale each layer in the stack:
   sc_water  <- scale_raster(preds_stack[["water"]],  means["water"], sds["water"])
   sc_temp   <- scale_raster(preds_stack[["temp"]],   means["temp"],  sds["temp"])
   sc_precip <- scale_raster(preds_stack[["precip"]], means["precip"],sds["precip"])
   sc_pop    <- scale_raster(preds_stack[["popdens"]],means["popdens"],sds["popdens"])
   # (drop landuse if categorical)

   scaled_stack <- c(sc_water, sc_temp, sc_precip, sc_pop)
   names(scaled_stack) <- c("water", "temp", "precip", "popdens")
   ```

2. **Compute PC scores**: multiply each scaled raster by the corresponding loading for PC1, then sum. E.g.:

   ```r
   # PC1 score raster = loading_water * sc_water + loading_temp * sc_temp + ...
   pc1_ras <- scaled_stack[["water"]]  * loadings["water", "PC1"] +
              scaled_stack[["temp"]]   * loadings["temp",  "PC1"] +
              scaled_stack[["precip"]] * loadings["precip","PC1"] +
              scaled_stack[["popdens"]] * loadings["popdens","PC1"]

   pc2_ras <- scaled_stack[["water"]]  * loadings["water", "PC2"] +
              scaled_stack[["temp"]]   * loadings["temp",  "PC2"] +
              scaled_stack[["precip"]] * loadings["precip","PC2"] +
              scaled_stack[["popdens"]] * loadings["popdens","PC2"]

   # And so on for PC3, PC4 if needed
   ```

3. **Create a new SpatRaster/Stack of PC scores**:

   ```r
   pc_stack <- c(pc1_ras, pc2_ras)   # (add pc3_ras, etc. if desired)
   names(pc_stack) <- c("PC1", "PC2")
   ```

   You can now feed `pc_stack` into any spatial model (GLM, RF, MaxEnt). Each PC is orthogonal, so no multicollinearity remains.

---

## 10. Final checklist

* [ ] **Check for missing values** in your extracted dataframe. Remove rows (pixels) with any `NA` before PCA.
* [ ] **Decide whether to keep categorical predictors**. Most PCA routines expect numeric data.
* [ ] **Document the centering/scaling parameters** (`means`, `sds`) so any new point or raster can be projected consistently.
* [ ] **Interpret loadings carefully**: variables with large absolute loadings on a PC drive that axis.
* [ ] If the goal is purely to remove multicollinearity, selecting representative variables (via correlation + loading) often yields more interpretable models than using PCs.
* [ ] If you go the PC‐as‐predictor route, be ready to explain what, e.g., “PC1“ means ecologically in your manuscript.

---

### Example summary of variable selection

Let's say your correlation matrix showed `(temp, precip) = 0.85`, `(water, precip) = 0.70`, `(popdens, others) < 0.4`. On PC1, loadings (absolute) were:

* precip = 0.61
* temp   = 0.56
* water  = 0.50
* popdens= 0.12

You might decide:

1. Keep **precip** (highest loading among \[temp, precip, water] on PC1).
2. Keep **popdens** (it's not highly correlated with others).
3. Either drop “water” or drop “temp” because they correlate with precip.
4. If there's another variable—say “soil moisture”—with loading 0.58 on PC2 and correlated with water, you could choose the one with the cleaner ecological interpretation.

Result: final predictors = `precip`, `popdens`, (and maybe one other uncorrelated variable).

---

#### In summary:

* Extract raster values → build a numeric data frame.
* Center & scale → run `prcomp()`.
* Inspect variance explained & loadings → pick either (A) PCs or (B) a reduced subset of original variables.
* Subset your raster stack or build PC rasters accordingly.
