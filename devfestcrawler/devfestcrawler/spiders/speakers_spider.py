# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'../')

import json
    
from scrapy.selector import Selector
from scrapy.spiders import Spider,CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from devfestcrawler.items import DevfestcrawlerItem


class SpeakersCrawler(CrawlSpider):
    name = "devfestcrawler"
    allowed_domains = ["https://devfest.pl"]
    start_urls = ["https://devfest.pl/speakers/"]
    
    rules = (
        Rule(LinkExtractor(allow=(), 
                               restrict_xpaths=('//div[@class="speaker"]',)
                               ), 
             callback="parse_items", 
             follow= True),
    )

    def parse_items(self, response):
        hxs = Selector(response)
        print hxs
        full_recipe = hxs.xpath('//div[@class="tags"]')
        print full_recipe

#        ingredients = full_recipe.xpath("//span[@itemprop = 'ingredients']/text()")        
#        ingr_list = []
#        item = CozinhabrasileiraItem()
#        for ingredient in ingredients:        
#            ingr_list.append(ingredient.extract())
            
        ingredients = full_recipe.xpath("//span[@itemprop = 'ingredients']/text()")
        ingr_list = []
        for ingredient in ingredients:        
            ingr_list.append(ingredient.extract())
        items = {"ingredients":ingr_list,
                 "recipe":full_recipe.xpath("//span[@itemprop = 'recipeInstructions']/text()").extract(),
                 "title":hxs.xpath("//header/h1[@itemprop = 'name headline']/text()").extract()
                }
        return items
#        item["ingredients"] = ingr_list
#        item["recipe"] = full_recipe.xpath("//span[@itemprop = 'recipeInstructions']/text()").extract()
#        item["title"] = hxs.xpath("//header/h1[@itemprop = 'name headline']/text()").extract()
#        items = []
#        items.append(item)
#        return(items)
       