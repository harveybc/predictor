<?php
$this->breadcrumbs=array(
	'Usuarios'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Gestionar'),
);

$this->menu=array(
		array('label'=>Yii::t('app',
				'Lista de Usuarios'), 'url'=>array('index')),
		array('label'=>Yii::t('app', 'Nuevo Usuario'),
				'url'=>array('create')),
			);

		Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('usuarios-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
		?>

<?php $this->setPageTitle (' Gestionar&nbsp;Usuarios'); ?>

<?php echo CHtml::link(Yii::t('app', 'Búsqueda Avanzada'),'#',array('class'=>'search-button')); ?>
<div class="search-form" style="display:none">
<?php $this->renderPartial('_search',array(
	'model'=>$model,
)); ?>
</div>

<?php $this->widget('zii.widgets.grid.CGridView', array(
	'id'=>'usuarios-grid',
	'dataProvider'=>$model->search(),
	'filter'=>$model,
        'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
	'columns'=>array(
		//'id',
		'Username',
		
		'Analista',
		'Proceso',
		'Es_administrador',
            'Es_analista',
		/*
		'esAdministrador',
                 'password',
		*/
		array(
			'class'=>'CButtonColumn',
		),
	),
)); ?>
