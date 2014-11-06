<?php
$this->pageTitle=Yii::app()->name . ' - Sitemap';
$this->breadcrumbs=array(
	'Sitemap',
);
?>
<?php $this->setPageTitle ('Mapa del Sitio'); ?>
<img src="/index.php/KPI_L1/dynamicImageGen" />
<div name="sbL1_img" id="sbL1_img">Espere...</div>
<div style="text-align: left;">
<?php




$dataTree=array(
		
    		array(
			'text'=>'<a href="/index.php/site/page?view=motoresElectricos"><b>Motores Eléctricos</b></a>', //must using 'text' key to show the text
			'children'=>array(//using 'children' key to indicate there are children
				array(
					'text'=>'<a href="/index.php/aislamiento_tierra/admin">Registro Aislamiento</a>',
					
				),
				array(
					'text'=>'<a href="/index.php/motores/admin">Base de Datos Motores</a>',
					
				),
				array(
					'text'=>'<a href="/index.php/vibraciones/admin">Registro Vibraciones y Temperatura</a>',
                                    
				),
                            array(
					'text'=>'<a href="/index.php/resumen/admin">Resumen de Resultados</a>' ,
                                    
				),
                            array(
					'text'=>'<a href="/index.php/tipo/admin">Lubricantes</a>',
                                    
				),
                                
			)
		),
    
    array(
			'text'=>'<a href="/index.php/site/page?view=reportes"><b>Reportes</b></a>', //must using 'text' key to show the text
			'children'=>array(//using 'children' key to indicate there are children
				array(
					'text'=>'<a href="/index.php/reportes/admin">Ultrasonido</a>',
					
				),
			      
			)
		),
    array(
			'text'=>'<a href="/index.php/site/page?view=termografia"><b>Termografía</b></a>', //must using 'text' key to show the text
			'children'=>array(//using 'children' key to indicate there are children
				array(
					'text'=>'<a href="/index.php/termotablero/admin">Informes de Tableros</a>',
					
				),
				array(
					'text'=>'<a href="/index.php/termomotores/admin">Motores</a>',
					
				),
				array(
					'text'=>'<a href="/index.php/tableros/admin">Tableros</a>',
                                    
				),
                                              
                                
			)
		),
    array(
			'text'=>'<a href="/index.php/aceitesnivel1/admin"><b>Aceites 1er Nivel</b></a>', //must using 'text' key to show the text
			'children'=>array(//using 'children' key to indicate there are children
				array(
					'text'=>'<a href="/index.php/aceitesnivel1/admin">Aceites 1er Nivel</a>',
					
				),
								                                                                                        
			)
	),
    
    
    array(
			'text'=>'<a href="/index.php/site/page?view=administrar"><b>Administrar</b></a>', //must using 'text' key to show the text
			'children'=>array(//using 'children' key to indicate there are children
				array(
					'text'=>'<a href="/index.php/site/page?view=tableroVirtualCbm">Tablero Virtual CBM</a>',
					
				),
				array(
					'text'=>'<a href="/index.php/usuarios/admin">Analistas</a>',
					
				),
				array(
					'text'=>'<a href="/index.php/estructura/admin">Equipos</a>',
                                    
				),
                            array(
					'text'=>'<a href="/index.php/site/page?view=buscarOT">Buscar por Orden de Trabajo</a>',
					
				),
                      
                    
                                
			)
          ),
    
    
    );

$this->widget('CTreeView',array(
        'data'=>$dataTree,
        'animated'=>'fast', //quick animation
        'collapsed'=>'false',//remember must giving quote for boolean value in here
        'htmlOptions'=>array(
                'class'=>'treeview-gray',//there are some classes that ready to use
        ),
));
?>
    </div>
<!--
<ul>
  <li>Coffee
    <ul>
          <li><a href="/index.php/Reportes/admin">Reportes</a></li>
          <li>Tea</li>
          <li>Milk</li>
    </ul>
  </li>
  <li>Tea</li>
  <li>Milk</li>
</ul>



array(
			'text'=>'Gramna', //must using 'text' key to show the text
			'children'=>array(//using 'children' key to indicate there are children
				array(
					'text'=>'Father',
					'children'=>array(
						array('text'=>'me'),
						array('text'=>'big sis'),
						array('text'=>'little brother'),
					)
				),
				array(
					'text'=>'Uncle',
					'children'=>array(
						array('text'=>'Ben'),
						array('text'=>'Sally'),
					)
				),
				array(
					'text'=>'Aunt',
				)
			)
		)
    
    
    );

--->
<script>
    //actualiza el Scoreboard al cargar la página
    actualiza_sb_img();
    // timer en javascript que ejecuta actualizaSB cada 9 segundos
    setInterval ( "actualiza_sb_img()", 6000 );
    //función que llama a acción KPI_L1/dynamicInfo con ajax y coloca el resultado en #sbL1
    function actualiza_sb_img( )// here is the magic
    {

<?php
echo CHtml::ajax(array(
    'url' => array('/KPI_L1/dynamicImage'),
    'type' => 'post',
    'update' => '#sbL1_img', //selector to update
));
?>
        return false;
    }
    
</script>  