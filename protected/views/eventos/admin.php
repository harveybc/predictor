<?php
$this->breadcrumbs=array(
	'Eventos'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Manage'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'Lista de Eventos'), 'url'=>array('index')),
		
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('eventos-grid', {
data: $(this).serialize()
});
				return false;
				});
			");

		?>

<?php $this->setPageTitle (' Gestionar&nbsp;Eventos'); ?>

<?php echo CHtml::link(Yii::t('app', 'Advanced Search'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'eventos-grid',
    //'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
	'dataProvider'=>$model->search(),
	'filter'=>$model,
	'columns'=>array(
		'id',
		'usuario',
		'modulo',
		//'operacion',
		'ip',
		'descripcion',
		
		'fecha',
		
		array(
			'class'=>'CButtonColumn',
                    'template'=>'{view}',
		),
	),
)); ?>
