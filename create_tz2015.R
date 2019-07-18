rm(list=ls())

tz <- read.table("./Data/00 Africa 1900-2015 SSA PR database (260617).tab",
                 header=TRUE)
tz <- tz[tz$COUNTRY=="Tanzania",]
tz <- tz[tz$YY==2015,]

plot(tz[,c("Long","Lat")],pch=20,asp=1)

library(sf)

tz <- st_as_sf(tz,coords=c("Long","Lat"))
st_crs(tz) <- 4236
tz <- st_transform(tz,crs=32736)


st_write(tz,'./Data/tz2015.shp')

tz <- st_read("./Data/tz2015.shp")

