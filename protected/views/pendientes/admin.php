<?php
$this->breadcrumbs=array(
	'Pendientes'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Manage'),
);
/*
 * 
 * 
$this->menu=array(
		array('label'=>Yii::t('app',
				'Lista de Pendientes'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Create Pendientes'),
				'url'=>array('create')),
			);
*/
		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('pendientes-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Manage&nbsp;Pendientes'); ?>

<?php echo CHtml::link(Yii::t('app', 'Advanced Search'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'pendientes-grid',
	'dataProvider'=>$model->search(),
	'filter'=>$model,
      //  'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
	'columns'=>array(
		'fecha_enviado',
                array(    
                    'name'=>'ruta',
                    'type'=>'raw',
                    'htmlOptions'=>array(
                        'width'=>'300px',
                    ),
                    'value'=>'CHtml::link($data->ruta,$data->ruta)',
                ),
		'usuario',
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
