<?php
$this->breadcrumbs=array(
	'Avisos ZI'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Gestionar'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'Lista de Avisos ZI'), 'url'=>array('index')),
		//array('label'=>Yii::t('app', 'Crear Aviso ZI'),				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('avisos-zi-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Gestionar&nbsp;Avisos ZI'); ?>

<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php 
// función que genera los enlaces
function genEnlace($data)
{
    if (isset($data))
        return('<a href="'.$data->Ruta.'">'.$data->Ruta.'</a>');
}

$this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'avisos-zi-grid',
	'dataProvider'=>$model->search(),
	'filter'=>$model,
                'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
	'columns'=>array(
		//'id',
            	'Fecha',
		//'Ruta',
            array(
            'header'=>'Ruta',
            'type'=>'raw',
            'value' => 'genEnlace($data)',
        ),
            //'Codigo',
		//'Operador',
		'Estado',
		//'Observaciones',
		//'arreglado',
		//'plan_mant',
		'OT',
		array(
			'class'=>'CButtonColumn',
                        'template'=>'{view}'
		),
	),
)); ?>
